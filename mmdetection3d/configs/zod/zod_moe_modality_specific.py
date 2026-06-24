"""Variant B — ModalitySpecificMoE on top of zod_bevfusion_dualinit (28 epoch).

Architecture (symmetric output-space modality-specific MoE)
-----------------------------------------------------------
::

    cam_bev (80 ch)  ─→ cam_summary  ─→ z_C ─┐
                                              ├─ concat → z_MC
    lidar_bev (256ch) → lidar_summary ─→ z_L ─┘     │
                                                    │ (.detach() for gate)
                          ┌─── gate ──── z_MC.detach() ┐
                          ▼                            ▼
                  full softmax (4 experts)        context_head
                          │                            │
                          │                        ctx_logits
                          ▼
        ┌────────────────────────────────────────────────────────────┐
        │   cam_direct   = cam_direct_proj(cam_bev)   # 80   → 256    │
        │   lidar_direct = lidar_bev                  # 256  used    │
        │                                                             │
        │   m_C = p_0 + p_1                                           │
        │   m_L = p_2 + p_3                                           │
        │   direct_mix = m_C · cam_direct + m_L · lidar_direct        │
        │                                                             │
        │   delta_sum  = Σ_(e∈cam)   p_e · (E_C(cam_direct)   − cam_direct)   │
        │              + Σ_(e∈lidar) p_e · (E_L(lidar_direct) − lidar_direct) │
        │                                                             │
        │   fused = refine(direct_mix + g · delta_sum)                │
        └────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                       pts_backbone → pts_neck → bbox_head

Cam direct proj:  (B,  80, H, W) → (B, 256, H, W)  (1×1 → BN → ReLU)
LiDAR direct:     (B, 256, H, W)  used as-is
Camera experts:   (B, 256, H, W) → (B, 256, H, W)  (BEVBottleneckResidualExpert)
LiDAR  experts:   (B, 256, H, W) → (B, 256, H, W)  (BEVBottleneckResidualExpert)
Fused output:     (B, 256, H, W)

Expert pools (4 experts total, dense flat routing):

    expert 0:  camera expert 0
    expert 1:  camera expert 1
    expert 2:  LiDAR  expert 0
    expert 3:  LiDAR  expert 1

Flat-routing guarantee
----------------------
There is still exactly one gate over E = num_cam_experts +
num_lidar_experts experts.  Modality-specificity is preserved at the
*expert input* level (camera experts only see cam_direct, LiDAR
experts only see lidar_direct) but all experts operate in the shared
256-channel output space.  This is **not** hierarchical routing —
there is no separate modality gate, no per-modality softmax, no
two-stage router.

Why symmetric output-space?
---------------------------
The old design (per-modality residual then concat → 1×1 → 3×3 fuser)
only controlled how much residual adaptation each modality received
but gave the routing no direct lever over the LiDAR/camera
contribution mix in the final fused BEV.  A LiDAR-anchored base path
would hard-code LiDAR as privileged; instead, this implementation
projects camera into the shared 256-channel width with a single 1×1
conv (``cam_direct_proj``) and uses LiDAR directly (no learned LiDAR
base path).  Both modalities then enter the fused BEV symmetrically:
the same flat gate weights both the direct modality contributions
(``m_C · cam_direct + m_L · lidar_direct``) and the routed expert
residual deltas.  Modality dominance is decided by the gate and the
detection gradients during training rather than at construction
time.

Reusing existing experts
------------------------
Both expert pools are built with the existing ``make_bev_experts``
factory and ``expert_type='bottleneck'`` (i.e.
:class:`BEVBottleneckResidualExpert`).  Because the inputs after the
direct projection are already 256-channel, the experts are
constructed with ``channels=out_channels`` — no new expert class is
introduced.

Identity-at-init contract
-------------------------
``BEVBottleneckResidualExpert`` has its final BN affine parameters
zero-initialised so the adapter branch emits an exact-zero residual
at step 0.  Both ``cam_direct`` and ``lidar_direct`` are post-ReLU
non-negative tensors, so ``expert(x) = ReLU(x) = x`` and the
per-expert ``delta = expert(x) − x = 0`` at step 0.  Consequently::

    fused_at_init ≈ refine(m_C · cam_direct + m_L · lidar_direct)

— a learned softmax mixture of the two direct modality features.

Group balance under the symmetric design
----------------------------------------
``group_balance_coef`` defaults to **0.004**: small enough that the
gate can still learn to favour the genuinely stronger modality, but
large enough to prevent early routing collapse to a single modality
group while both direct paths are still adapting their BN statistics.
``cam_group_mass`` / ``lidar_group_mass`` measure the routed modality
contribution mass (and they also weight the direct mix, so they are
the dominant statistic for who-contributes-what to the fused BEV).

Auxiliary losses entering the optimisation total
------------------------------------------------
* importance_coef · importance_loss
* load_coef · load_loss                (0 under dense routing)
* z_loss_coef · router_z_loss
* group_balance_coef · group_balance_loss   (0.004 by default here)
* ctx_loss_coef · CE_weighted(ctx_logits, road_type)
"""
_base_ = ['./zod_bevfusion_dualinit_28ep.py']

# ─────────────────────────────────────────────────────────────────────────
# MoE block — 2 cam + 2 lidar dense experts replacing ConvFuser.
# Symmetric output-space design: camera is projected from 80 to the
# shared 256-channel width with ``cam_direct_proj``; LiDAR is used
# directly (no LiDAR base path).  Both expert pools reuse the existing
# BEVBottleneckResidualExpert at ``channels=out_channels=256``.  A
# single flat gate weights both the direct mix and the routed deltas.
# Routing hyperparameters mirror LiDAR-only MoE 4577584.
# ─────────────────────────────────────────────────────────────────────────
num_cam_experts   = 2
num_lidar_experts = 2
num_experts       = num_cam_experts + num_lidar_experts   # 4

modality_specific_moe_cfg = dict(
    type='ModalitySpecificMoEBlock',
    cam_channels=80,
    lidar_channels=256,
    out_channels=256,
    num_cam_experts=num_cam_experts,
    num_lidar_experts=num_lidar_experts,
    # Reuse the existing bottleneck expert; both pools are constructed
    # at channels=out_channels because cam_direct_proj brings camera
    # into the 256-channel shared space first.
    expert_type='full',
    # Dense routing — gate_type='dense' forces k=num_experts internally.
    gate_type='dense',
    gate_cfg=dict(temperature=1.0),
    gate_input_detach=True,
    # Routing + auxiliary loss coefficients copied from 4577584.
    importance_coef=0.005,
    load_coef=0.0,
    z_loss_coef=0.002,
    # Group balance ACTIVE under the symmetric design — small coefficient
    # to discourage early collapse of routing mass to a single modality
    # group while still letting the gate drift toward the genuinely
    # stronger modality if needed.
    group_balance_coef=0.004,
    residual_gain=1.0,
    # Per-modality summary: 128-d each → 256-d concat (matches 4577584
    # gate input dim).
    router_out_dim=128,
    # Context supervision on road_type with weighted CE + inverse
    # frequency + label smoothing — identical to 4577584.
    context_aux_cfg=dict(
        target_field='road_type',
        loss_coef=0.03,
        loss_type='weighted_ce',
        class_weights='inverse_frequency',
        label_smoothing=0.05,
    ),
)

# ─────────────────────────────────────────────────────────────────────────
# Model — remove ConvFuser; route through modality-specific experts.
# ─────────────────────────────────────────────────────────────────────────
model = dict(
    fusion_layer=None,
    modality_specific_moe_cfg=modality_specific_moe_cfg,
)

model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')

# ─────────────────────────────────────────────────────────────────────────
# Optimiser + LR schedule — copy lidar_only_moe 4577584, compressed to
# 28 epochs.  See zod_moe_fusion_then.py for the rationale.
# paramwise_cfg keys are adapted to the modality_specific_moe module path.
# ─────────────────────────────────────────────────────────────────────────
lr = 5e-5

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head':                              dict(lr_mult=1.0),
            'modality_specific_moe.context_head':     dict(decay_mult=0.05, lr_mult=1.0),
            'modality_specific_moe.cam_summary':      dict(decay_mult=0.05, lr_mult=1.0),
            'modality_specific_moe.lidar_summary':    dict(decay_mult=0.05, lr_mult=1.0),
            'modality_specific_moe.gate':             dict(decay_mult=0.01, lr_mult=1.0),
            'pts_backbone':                           dict(lr_mult=1.0),
            'pts_neck':                               dict(lr_mult=1.0),
        }),
    accumulative_counts=2,
)

# Converge-to-plateau schedule: 0–8 epoch LR ramp, 8–75 epoch cosine decay.
# Max 75 epochs; early stopping (patience 15, min 40 epochs) halts the run once the
# validation AP@50 plateaus, so the cosine tail is rarely fully traversed.
param_scheduler = [
    # 500-iter linear warmup (start_factor=1/3 → lr goes 5e-5/3 → 5e-5)
    dict(type='LinearLR',
         start_factor=0.33333333, begin=0, end=500,
         by_epoch=False),
    # Cosine ramp-up: 5e-5 → 5e-4 over epochs 0-8
    dict(type='CosineAnnealingLR',
         T_max=8, begin=0, end=8,
         eta_min=5e-4, by_epoch=True, convert_to_iter_based=True),
    # Cosine decay: 5e-4 → 5e-9 over epochs 8-75
    dict(type='CosineAnnealingLR',
         T_max=67, begin=8, end=75,
         eta_min=5e-9, by_epoch=True, convert_to_iter_based=True),
    # Coupled momentum annealing
    dict(type='CosineAnnealingMomentum',
         T_max=8, begin=0, end=8,
         eta_min=0.8947368421052632, by_epoch=True,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum',
         T_max=67, begin=8, end=75,
         eta_min=1, by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=75, val_interval=1)

# ── Best-checkpoint + early stopping ───────────────────────────────────────
# Validate every epoch (val_interval=1 above); keep the best checkpoint by
# AP@50 IoU (mAP_0.50); stop early once mAP_0.50 fails to improve by at least
# min_delta for `patience` consecutive validations.  Early stopping is
# disabled for the first min_epochs epochs to ensure a minimum training run.
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='mAP_0.50',
        rule='greater'),
    early_stopping=dict(
        type='MinEpochEarlyStoppingHook',
        monitor='mAP_0.50',
        rule='greater',
        patience=15,
        min_delta=0.001,
        min_epochs=40),
)

# ─────────────────────────────────────────────────────────────────────────
# Pipelines — base config + 'context' meta key, identical otherwise.
# ─────────────────────────────────────────────────────────────────────────
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]
class_names = ['pedestrian']
backend_args = None

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=1,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[704, 1248],
        resize_lim=[1.0, 1.0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[0, 0],
        translation_std=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='GridMask',
        use_h=True, use_w=True, max_epoch=28, rotate=1, offset=False,
        ratio=0.5, mode=1, prob=0.0, fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
            'lidar_aug_matrix', 'num_pts_feats',
            'context',
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=1,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[704, 1248],
        resize_lim=[1.0, 1.0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats',
            'context',
        ])
]

train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader  = dict(dataset=dict(pipeline=test_pipeline))

# ─────────────────────────────────────────────────────────────────────────
# Hooks — preserve base set + MoE diagnostic + respawn hooks.
# enable_hook_c=True: track camera-group / LiDAR-group mass per epoch.
# ─────────────────────────────────────────────────────────────────────────
_VIS_EPOCHS = (1, 5, 10, 15, 20, 25, 28)

custom_hooks = [
    dict(
        type='DualCheckpointInitHook',
        lidar_ckpt=(
            '/home/users/u103958/projects/multimodal-MoE/outputs/runs/'
            'zod_lidar_only/zod-lidar-only_4570893/'
            'best_mAP_0.50_epoch_30.pth'),
        camera_ckpt=(
            '/home/users/u103958/projects/multimodal-MoE/outputs/runs/'
            'zod_camera_only/zod-cam-only_4577582/'
            'best_mAP_0.50_epoch_31.pth'),
        lidar_modules=[
            'pts_middle_encoder', 'pts_backbone', 'pts_neck', 'bbox_head',
        ],
        camera_modules=[
            'img_backbone', 'img_neck', 'view_transform',
        ]),
    dict(type='BEVFeatureVisualizationHook',         vis_epochs=_VIS_EPOCHS),
    dict(type='BEVPredictionVisualizationHook',      score_thr=0.15,
         vis_epochs=_VIS_EPOCHS),
    dict(type='BEVValPredictionVisualizationHook',   score_thr=0.15,
         vis_epochs=_VIS_EPOCHS),
    dict(type='BEVCameraFeatureVisualizationHook',   vis_epochs=_VIS_EPOCHS),
    dict(type='DepthTransformDiagnosticHook',        vis_epochs=_VIS_EPOCHS),
    dict(type='TrainingEfficiencyHook'),
    dict(type='RunSummaryHook'),
    dict(type='DepthProjectionDebugHook',            vis_epochs=_VIS_EPOCHS),
    dict(type='ValidationCurveHook',
         metric_keys=('mAP_0.50', 'mAP_0.5m'),
         filename='val_curve_ap_0_50_0_5m'),
    # ── MoE-specific hooks (copied from 4577584) ───────────────────────
    dict(
        type='ExpertRespawnHook',
        num_experts=num_experts,
        dead_threshold_ratio=0.1,
        perturbation_std=0.02,
        max_respawns=5,
        skip_first_epoch=True,
    ),
    dict(
        type='MoERoutingHook',
        num_experts=num_experts,
        enable_hook_a=True,
        enable_hook_b=True,
        # Track camera vs LiDAR group mass — only meaningful here.
        enable_hook_c=True,
        ap_metric_key='mAP_0.5m',
    ),
    dict(type='ContextRoutingStatsHook'),
    dict(type='ContextExpertUsageVisualizationHook'),
]
