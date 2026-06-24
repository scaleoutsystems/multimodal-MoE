"""Variant C — Fusion-then-MoE on top of zod_bevfusion_dualinit (28 epoch).

Architecture
------------
::

    cam_bev (80 ch)  ─┐
                       ├─ ConvFuser ─→ fused_bev (256 ch)
    lidar_bev (256ch) ─┘                       │
                                                ▼
                                          pts_backbone
                                                │
                                                ▼
                                       pts_neck / SECONDFPN (out=512 ch)
                                                │
                                                ▼
                                     BEVMoEBlock (post-neck, 512 ch)
                                                │
                                                ▼
                                          TransFusionHead

Strict design copy
------------------
The MoE configuration here exactly mirrors the stable LiDAR-only MoE
40-epoch run 4577584 (dense soft-MoE, 4 experts, context-supervised
routing on ``road_type`` with weighted CE / inverse-frequency / label
smoothing).  The only differences from that run are the model
configuration — keep ``ConvFuser`` so cam+lidar BEVs are fused before
the backbone — and the training horizon (28 epochs instead of 40, to
fit two Variant-C runs in the 48 h Meluxina wall-time budget).

Insertion point: ``bev_moe_position='post_neck'`` (after SECONDFPN,
just before the TransFusionHead).  BEV channel count after SECONDFPN
is 512 (in_channels=[128, 256] → out_channels=[256, 256] →
concatenated 512), matching the LiDAR-only MoE configuration.

Auxiliary losses entering the optimisation total
------------------------------------------------
* importance_coef · importance_loss         (Shazeer)
* load_coef · load_loss                     (0 under dense routing)
* z_loss_coef · router_z_loss               (clean-logit log-Z²)
* switch_balance_coef · switch_balance_loss (0)
* ctx_loss_coef · CE_weighted(ctx_logits, road_type)
                                            (inverse frequency, ε-smooth)

No group_balance_loss for this variant — the MoE block sees a single
post-fusion 512-channel BEV with no modality grouping.
"""
_base_ = ['./zod_bevfusion_dualinit_28ep.py']

# ─────────────────────────────────────────────────────────────────────────
# MoE block — copy lidar_only_moe 4577584's bev_moe_cfg verbatim.
# Channel count is 512 (post-SECONDFPN output), which matches both
# variants.
# ─────────────────────────────────────────────────────────────────────────
num_experts = 4

bev_moe_cfg = dict(
    type='BEVMoEBlock',
    # SECONDFPN concatenates two 256-ch streams → 512 channels post-neck.
    # Matches the LiDAR-only MoE 4577584 channel count.
    channels=512,
    num_experts=num_experts,
    # Dense Soft-MoE: every expert always runs.  k is forced to
    # num_experts internally; keeping the explicit value mirrors the
    # 4577584 config exactly.
    k=num_experts,
    num_convs=2,
    expert_type='full',
    # GroupNorm inside the experts instead of (fp32-safe) BatchNorm.
    # Fusion-then-MoE operates on the fused (camera ⊕ lidar) BEV,
    # whose per-channel distribution has narrow modes from camera
    # depth-splat shells.  Under that input the expert BN
    # ``running_var`` drifts to ~1e-5 over ~5 epochs, after which the
    # BN output saturates fp16 in the next conv and TransFusionHead
    # softmax produces NaN — observed in runs 4610584 and 4613034
    # (epoch 6 iter 2550, see run notes).  GroupNorm has no running
    # stats and computes the denominator from per-sample current
    # activations, so this failure mode is removed by construction.
    # Lidar-only MoE keeps ``expert_norm_type='bn'`` (default) since
    # its input distribution does not drive the same degeneracy
    # (run 4610582 finished 30 epochs cleanly with BN experts).
    expert_norm_type='gn',
    gate_type='dense',
    gate_cfg=dict(temperature=1.0),
    gate_input_detach=True,
    # Routing + auxiliary loss coefficients, copied from 4577584.
    importance_coef=0.005,
    load_coef=0.0,
    z_loss_coef=0.002,
    switch_balance_coef=0.0,
    residual_gain=1.0,
    # No temperature warmup — 4577584 uses ctx_gate_warmup_epochs=0.
    ctx_gate_warmup_epochs=0,
    ctx_gate_temp_high=1.0,
    # Context supervision on road_type with weighted CE + inverse-
    # frequency class weights + label smoothing (copy of 4577584).
    context_aux_cfg=dict(
        target_field='road_type',
        loss_coef=0.03,
        loss_type='weighted_ce',
        class_weights='inverse_frequency',
        label_smoothing=0.05,
    ),
)

# ─────────────────────────────────────────────────────────────────────────
# Model — extend the base with bev_moe_cfg + bev_moe_position.
# fusion_layer (ConvFuser) is inherited from the base and KEPT — this is
# the only multimodal variant that retains ConvFuser.
# ─────────────────────────────────────────────────────────────────────────
model = dict(
    bev_moe_cfg=bev_moe_cfg,
    bev_moe_position='post_neck',
)

model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')

# ─────────────────────────────────────────────────────────────────────────
# Optimiser + LR schedule — copy lidar_only_moe 4577584, compressed to
# 28 epochs.  Original schedule: 8-epoch cosine warm-up to lr=5e-4, then
# 32-epoch cosine decay to ~5e-9.  Compressed: 8-epoch warm-up unchanged,
# 20-epoch cosine decay (8 → 28).  Same applies to momentum.
# Budget was originally cut to 24 ep when the full-channel experts forced
# bs=2+accum=2 and threatened the 48 h wall-time; the new bottleneck
# experts (~6× cheaper) let us restore the planned 28-epoch schedule with
# comfortable margin.
# AmpOptimWrapper, AdamW (lr=5e-5, wd=0.01), clip_grad max_norm=10.
# paramwise_cfg lowers weight decay on context-summary / context-head /
# gate so the routing branch can adapt fast without ramping into the
# default L2.
# ─────────────────────────────────────────────────────────────────────────
lr = 5e-5

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head':               dict(lr_mult=1.0),
            'bev_moe.context_head':    dict(decay_mult=0.05, lr_mult=1.0),
            'bev_moe.context_summary': dict(decay_mult=0.05, lr_mult=1.0),
            'bev_moe.gate':            dict(decay_mult=0.01, lr_mult=1.0),
            'pts_backbone':            dict(lr_mult=1.0),
            'pts_neck':                dict(lr_mult=1.0),
        }),
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
# Pipelines — base config + 'context' meta key for the MoE block to
# resolve road_type labels.  Everything else is preserved exactly.
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

# Reduce per-GPU batch size from 4 → 2 to fit the larger fusion+MoE graph
# (BEVFusion + BEVMoEBlock OOMs at batch=4 on 40 GB A100s).
# Gradient accumulation of 2 keeps effective global batch at 4 GPU × 2 × 2 = 16,
# matching the dualinit baseline.
train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader  = dict(dataset=dict(pipeline=test_pipeline))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    # bf16 instead of the default fp16.  The fusion-then variant
    # combines camera + lidar gradients through the shared backbone and
    # neck, which after ~10 epochs of combined gradient pressure pushes
    # some activation into fp16 overflow (max representable ~65504).
    # bf16 has 8-bit exponent (same dynamic range as fp32,
    # max ~3.4×10^38), so overflow is impossible in practice.
    # Accuracy is the same as fp16 (both have 7-8 mantissa bits for
    # the activation values that matter here).
    # Confirmed sufficient: runs 4610584, 4613034, 4615918 all
    # collapsed via fp16 overflow at epoch 5-10; no change to
    # architecture, losses, or schedules needed.
    dtype='bfloat16',
    accumulative_counts=2,
)

# ─────────────────────────────────────────────────────────────────────────
# Hooks — preserve the base visualisation/diagnostic hooks (incl. the
# camera-branch-specific BEVCameraFeatureVisualizationHook and depth
# diagnostics that don't exist in the LiDAR-only-MoE run) and add the
# MoE-specific hooks copied from 4577584.
# ─────────────────────────────────────────────────────────────────────────
_VIS_EPOCHS = (1, 5, 10, 15, 20, 25, 28)

custom_hooks = [
    dict(
        type='DualCheckpointInitHook',
        lidar_ckpt=(
            '/home/users/u103958/projects/multimodal-MoE/outputs/runs/zod_lidar_only/zod-lidar-only_4570893/best_mAP_0.50_epoch_30.pth'),
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
    # ── MoE-specific diagnostic + maintenance hooks (copied from 4577584).
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
        # No modality grouping on the post-fusion BEV → no group mass.
        enable_hook_c=False,
        ap_metric_key='mAP_0.5m',
    ),
    dict(type='ContextRoutingStatsHook'),
    dict(type='ContextExpertUsageVisualizationHook'),
]
