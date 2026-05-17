"""Variant A — JointModalityMoE on top of zod_bevfusion_dualinit (28 epoch).

Architecture
------------
::

    cam_bev (80 ch) ──→ cam_summary  ─→ z_C ─┐
                                              ├─→ concat → z_LC
    lidar_bev (256ch) → lidar_summary ─→ z_L ─┘     │
                                                    │ (.detach() for gate)
                          ┌─────── gate ──── z_LC.detach() ┐
                          ▼                                ▼
                     full softmax (4 experts)         context_head
                          │                                │
                          │                            ctx_logits
                          ▼                                │
              (cam_bev, lidar_bev) ──→ JointModalityMoEBlock
                                          │
                                          ▼
                          dense mix of 4 joint experts
                                          │
                                          ▼
                                   fused_bev (256 ch)
                                          │
                                          ▼
                       pts_backbone → pts_neck → bbox_head

Strict design copy
------------------
Routing is identical to LiDAR-only MoE run 4577584 (dense soft-MoE,
4 experts, context-supervised routing on ``road_type`` with weighted CE
+ inverse frequency + label smoothing).  ConvFuser is REMOVED — each
joint expert is itself a fresh-fusion module
(concat([cam,lidar]) → 3×3 conv → BN → ReLU → 256-ch fused BEV).  The
dense gate forms a convex mixture across the four joint experts.

Strict delta vs ``zod_bevfusion_dualinit_28ep.py``
--------------------------------------------------
* dataset / dataloaders                        — UNCHANGED (only
  ``Pack3DDetInputs.meta_keys`` is extended with ``'context'``).
* dual-init logic / DualCheckpointInitHook    — UNCHANGED.
* backbone / neck / bbox_head                 — UNCHANGED.
* model.fusion_layer                          — REMOVED (None).
* Added:  ``model.joint_modality_moe_cfg``     (4 dense experts).
* Replaced: optim_wrapper + param_scheduler   (copied from 4577584,
  compressed to 28 epochs).
* Added:  MoE diagnostic + respawn hooks.

Auxiliary losses entering the optimisation total
------------------------------------------------
* importance_coef · importance_loss
* load_coef · load_loss              (0 under dense routing)
* z_loss_coef · router_z_loss
* ctx_loss_coef · CE_weighted(ctx_logits, road_type)

No group_balance_loss for this variant — all experts share a single
joint-modality pool.
"""
_base_ = ['./zod_bevfusion_dualinit_28ep.py']

# ─────────────────────────────────────────────────────────────────────────
# MoE block — 4 dense joint-modality experts replacing ConvFuser.
# Routing hyperparameters mirror LiDAR-only MoE 4577584 exactly.
# ─────────────────────────────────────────────────────────────────────────
num_experts = 4

joint_modality_moe_cfg = dict(
    type='JointModalityMoEBlock',
    cam_channels=80,
    lidar_channels=256,
    out_channels=256,
    num_experts=num_experts,
    # Dense routing — gate_type='dense' forces k=num_experts internally.
    gate_type='dense',
    gate_cfg=dict(temperature=1.0),
    gate_input_detach=True,
    # Routing + auxiliary loss coefficients copied from 4577584.
    importance_coef=0.005,
    load_coef=0.0,
    z_loss_coef=0.002,
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
# Model — remove ConvFuser; route through the joint-modality MoE block.
# ─────────────────────────────────────────────────────────────────────────
model = dict(
    fusion_layer=None,
    joint_modality_moe_cfg=joint_modality_moe_cfg,
)

model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')

# ─────────────────────────────────────────────────────────────────────────
# Optimiser + LR schedule — copy lidar_only_moe 4577584, compressed to
# 28 epochs.  See zod_moe_fusion_then.py for the rationale.
# paramwise_cfg keys are adapted to the joint_modality_moe module path.
# ─────────────────────────────────────────────────────────────────────────
lr = 5e-5

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head':                            dict(lr_mult=1.0),
            'joint_modality_moe.context_head':      dict(decay_mult=0.05, lr_mult=1.0),
            'joint_modality_moe.cam_summary':       dict(decay_mult=0.05, lr_mult=1.0),
            'joint_modality_moe.lidar_summary':     dict(decay_mult=0.05, lr_mult=1.0),
            'joint_modality_moe.gate':              dict(decay_mult=0.01, lr_mult=1.0),
            'pts_backbone':                         dict(lr_mult=1.0),
            'pts_neck':                             dict(lr_mult=1.0),
        }),
    accumulative_counts=2,
)

param_scheduler = [
    dict(type='CosineAnnealingLR',
         T_max=8, begin=0, end=8,
         eta_min=5e-4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR',
         T_max=20, begin=8, end=28,
         eta_min=5e-9, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum',
         T_max=8, begin=0, end=8,
         eta_min=0.8947368421052632, by_epoch=True,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum',
         T_max=20, begin=8, end=28,
         eta_min=1, by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=28, val_interval=1)

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
            'zod_camera_only/zod-cam-only_4469392/'
            'best_mAP_0.50_epoch_11.pth'),
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
        # No modality-group expert pools in the joint-modality block.
        enable_hook_c=False,
        ap_metric_key='mAP_0.5m',
    ),
    dict(type='ContextRoutingStatsHook'),
    dict(type='ContextExpertUsageVisualizationHook'),
]
