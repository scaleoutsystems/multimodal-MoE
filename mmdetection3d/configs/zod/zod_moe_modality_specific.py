"""Variant B — ModalitySpecificMoE on top of zod_bevfusion_dualinit.

Architecture
------------
::

    cam_bev (80 ch)  ─→ cam_summary  ─→ z_C ─┐
                                              ├─ concat ─→ gate (over joint pool)
    lidar_bev (256ch) → lidar_summary ─→ z_L ─┘                       │
                                                                       ▼
        cam_bev   → cam_experts[0..2]  ─┐                  top-k expert dispatch
                                         ├─→ concat → 1×1 → 3×3 → fused (256 ch)
        lidar_bev → lidar_experts[0..2] ─┘                              │
                                                                        ▼
                                              pts_backbone → pts_neck → bbox_head

Two dedicated expert pools (3 camera, 3 LiDAR) with a single joint gate
of size ``E = 6``.  Top-2 over the joint pool lets the router pick

    LiDAR + LiDAR    Camera + Camera    LiDAR + Camera

combinations.  ``ConvFuser`` is removed; the block performs its own
two-step concat → 1×1 → 3×3 fusion after expert dispatch.

Routing
~~~~~~~
Same descriptor-level conditioning as Variant A::

    z = concat(z_C, z_L)              # (B, 256)
    gate(z) → top-k over the joint 6-expert pool
    context_head(z) → CE on road_type

Group balance loss
~~~~~~~~~~~~~~~~~~
This is the only variant that uses ``group_balance_loss`` — it
penalises imbalance between the camera and LiDAR groups in the soft
router belief, preventing the router from collapsing onto a single
modality's experts.

Strict delta vs ``zod_bevfusion_dualinit.py``
---------------------------------------------
* dataset / dataloaders                        — UNCHANGED (only
  ``Pack3DDetInputs.meta_keys`` is extended with ``'context'``).
* training schedule / optimizer                — UNCHANGED.
* dual-init logic / DualCheckpointInitHook    — UNCHANGED.
* backbone / neck definitions / bbox_head     — UNCHANGED.
* model.fusion_layer                          — REMOVED (set to None);
  modality_specific_moe performs its own fusion via concat + 1×1 + 3×3.
* Added: ``model.modality_specific_moe_cfg`` (3 cam + 3 lidar experts,
  top-2, group_balance_coef=0.01, context-supervised on ``road_type``).
* Added: MoE diagnostic hooks (with enable_hook_c=True for
  modality-group mass tracking) + ExpertRespawnHook.
* Added: ``model_wrapper_cfg = dict(find_unused_parameters=True)``.

Auxiliary losses entering the optimisation total
------------------------------------------------
* importance_loss
* load_loss            (NoisyTopkGate; only non-zero in train mode)
* router_z_loss
* group_balance_loss   (cam vs lidar group mass balance)
* ctx_aux_loss_weighted = ctx_aux_coef · CE(ctx_logits, road_type)

Expert IDs
----------
The block hard-codes::

    cam_expert_ids   = [0, 1, 2]
    lidar_expert_ids = [3, 4, 5]

These are derived from ``num_cam_experts=3`` / ``num_lidar_experts=3``
inside ``ModalitySpecificMoEBlock.__init__`` and are exposed via
``moe_info['cam_expert_ids']`` / ``moe_info['lidar_expert_ids']`` so
the downstream hooks (MoERoutingHook with ``enable_hook_c=True``) can
group routing statistics by modality.
"""
_base_ = ['./zod_bevfusion_dualinit.py']

# ─────────────────────────────────────────────────────────────────────────
# MoE block — separate cam/lidar expert pools with a joint gate.
# ─────────────────────────────────────────────────────────────────────────
num_cam_experts   = 3
num_lidar_experts = 3
num_experts       = num_cam_experts + num_lidar_experts   # 6

modality_specific_moe_cfg = dict(
    type='ModalitySpecificMoEBlock',
    cam_channels=80,
    lidar_channels=256,
    out_channels=256,
    num_cam_experts=num_cam_experts,
    num_lidar_experts=num_lidar_experts,
    k=2,
    num_convs=1,
    importance_coef=0.02,
    load_coef=0.002,
    z_loss_coef=1e-4,
    # Camera-vs-LiDAR group-balance penalty on the soft router belief.
    # Only this variant emits group_balance_loss into aux_loss.
    group_balance_coef=0.01,
    residual_gain=1.0,
    router_pool_size=4,
    router_spatial_dim=128,
    router_hidden_dim=256,
    router_out_dim=128,
    context_aux_cfg=dict(
        target_field='road_type',
        loss_coef=0.05,
        label_smoothing=0.0,
    ),
    gate_type='noisy_topk',
    gate_cfg=dict(noise_epsilon=1e-2, temperature=1.0),
)

# ─────────────────────────────────────────────────────────────────────────
# Model — remove ConvFuser; route through modality-specific experts.
# ─────────────────────────────────────────────────────────────────────────
model = dict(
    fusion_layer=None,
    modality_specific_moe_cfg=modality_specific_moe_cfg,
)

model_wrapper_cfg = dict(find_unused_parameters=True)

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
        use_h=True, use_w=True, max_epoch=10, rotate=1, offset=False,
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader  = dict(dataset=dict(pipeline=test_pipeline))

# ─────────────────────────────────────────────────────────────────────────
# Hooks — preserve base set + MoE diagnostic + respawn hooks.
# enable_hook_c=True: track camera-group / LiDAR-group mass per epoch.
# ─────────────────────────────────────────────────────────────────────────
_VIS_EPOCHS = (1, 3, 5, 7, 10, 12)

custom_hooks = [
    dict(
        type='DualCheckpointInitHook',
        lidar_ckpt=(
            '/home/users/u103958/projects/multimodal-MoE/outputs/runs/'
            'zod_lidar_only/zod-lidar-only_4454825/'
            'best_mAP_0.50_epoch_18.pth'),
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
    # ── MoE-specific hooks ─────────────────────────────────────────────
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
