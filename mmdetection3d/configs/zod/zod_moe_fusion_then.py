"""Variant C — Fusion-then-MoE on top of zod_bevfusion_dualinit.

Architecture
------------
::

    cam_bev (80 ch)  ─┐
                       ├─ ConvFuser ─→ fused_bev (256 ch)
    lidar_bev (256ch) ─┘                       │
                                                ▼
                                        BEVMoEBlock (single)
                                                │
                                                ▼
                                          pts_backbone → pts_neck → bbox_head

This is the only MoE variant that keeps ``ConvFuser``.  The MoE block
operates on the post-fusion 256-channel BEV; experts share a single
pool with no modality grouping and therefore no group_balance_loss.

Strict delta vs ``zod_bevfusion_dualinit.py``
---------------------------------------------
* dataset / dataloaders                        — UNCHANGED (only
  ``Pack3DDetInputs.meta_keys`` is extended with ``'context'`` so the
  MoE block can read the road_type label from ``batch_input_metas``).
* training schedule / optimizer                — UNCHANGED.
* dual-init logic / DualCheckpointInitHook    — UNCHANGED (fusion_layer
  remains randomly initialised — same as the base config).
* backbone / neck definitions / bbox_head     — UNCHANGED.
* model.fusion_layer (ConvFuser)              — UNCHANGED (kept).
* Added: ``model.bev_moe_cfg`` (BEVMoEBlock with 5 experts, top-2,
  context-supervised routing on ``road_type``).
* Added: MoE diagnostic hooks (MoERoutingHook, ContextRoutingStatsHook,
  ContextExpertUsageVisualizationHook, ExpertRespawnHook).
* Added: ``model_wrapper_cfg = dict(find_unused_parameters=True)`` —
  required for DDP under top-k sparse expert dispatch.

Auxiliary losses entering the optimisation total
------------------------------------------------
* importance_loss      (Shazeer)
* load_loss            (Shazeer; only non-zero when NoisyTopkGate is
                        in training mode)
* router_z_loss        (z_loss_coef · log-Z² over clean_logits)
* ctx_aux_loss_weighted = ctx_aux_coef · CE(ctx_logits, road_type)

No group_balance_loss for this variant — there is no modality grouping
on the post-fusion BEV.
"""
_base_ = ['./zod_bevfusion_dualinit.py']

# ─────────────────────────────────────────────────────────────────────────
# MoE block — single pool of 5 residual-conv experts on the fused BEV.
# ─────────────────────────────────────────────────────────────────────────
num_experts = 5

bev_moe_cfg = dict(
    type='BEVMoEBlock',
    # ConvFuser outputs 256 channels (in_channels=[80, 256], out=256), so
    # the MoE block runs on 256-channel BEV with input == output channels.
    channels=256,
    num_experts=num_experts,
    k=2,
    num_convs=1,
    # Shazeer auxiliary loss weights (matched to zod_lidar_only_moe).
    importance_coef=0.02,
    load_coef=0.002,
    # Mesh-Transformer / ST-MoE router z-regulariser on clean_logits.
    z_loss_coef=1e-4,
    residual_gain=1.0,
    # BEVSummaryHead defaults — gate input dim = router_out_dim (no
    # context concatenation).
    router_pool_size=4,
    router_spatial_dim=128,
    router_hidden_dim=256,
    router_out_dim=128,
    # Context supervision via auxiliary CE on road_type.  Loss only
    # shapes the BEV summary descriptor; it never enters the gate input.
    context_aux_cfg=dict(
        target_field='road_type',
        loss_coef=0.05,
        label_smoothing=0.0,
    ),
    # NoisyTopkGate: load_loss is only computable when noise_std is
    # populated (training time only).  Eval forwards return noise_std=None
    # and load_loss returns 0 — exactly the requested behaviour.
    gate_type='noisy_topk',
    gate_cfg=dict(noise_epsilon=1e-2, temperature=1.0),
)

# ─────────────────────────────────────────────────────────────────────────
# Model — extend the base with bev_moe_cfg; keep fusion_layer (ConvFuser).
# ─────────────────────────────────────────────────────────────────────────
model = dict(bev_moe_cfg=bev_moe_cfg)

# DDP under top-k sparse routing leaves idle experts without gradient.
# Tell DDP to tolerate unused params instead of crashing on iter 2.
model_wrapper_cfg = dict(find_unused_parameters=True)

# ─────────────────────────────────────────────────────────────────────────
# Pipelines — identical to the base except 'context' is added to
# Pack3DDetInputs.meta_keys so BEVMoEBlock can resolve road_type labels
# from batch_input_metas (extract_context_labels reads meta['context']).
# Everything else (augmentations, ranges, modes) is preserved exactly.
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

# Re-bind the pipeline lists into each dataloader's dataset config.
# mmengine merge: list values overwrite atomically, so only `pipeline`
# is replaced; data_root, ann_file, etc. are inherited from the base.
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader  = dict(dataset=dict(pipeline=test_pipeline))

# ─────────────────────────────────────────────────────────────────────────
# Hooks — preserve the full set from the base config and append MoE
# diagnostic + respawn hooks.  Lists are atomic in mmengine merging, so
# this list fully replaces the base list (we copy every entry verbatim).
# ─────────────────────────────────────────────────────────────────────────
_VIS_EPOCHS = (1, 3, 5, 7, 10, 12)

custom_hooks = [
    # Dual-checkpoint init — UNCHANGED from base.  fusion_layer (ConvFuser)
    # and the MoE block both initialise from random; everything else
    # warm-starts as in the dual-init baseline.
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
    # ── Base visualisation / diagnostic hooks (unchanged) ──────────────
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
    # ── MoE-specific diagnostic + maintenance hooks ────────────────────
    # ExpertRespawnHook fires before MoERoutingHook each end-of-epoch
    # (priorities NORMAL=50 < BELOW_NORMAL=60), so the next epoch's
    # routing plot already reflects any respawn surgery.
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
