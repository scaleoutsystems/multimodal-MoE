"""Camera + LiDAR BEVFusion — controlled dual-init baseline, finetune-matched schedule.

Derived from zod_bevfusion_finetune.py.  Only the settings listed below
differ from the base finetune config; everything else (model architecture,
data pipelines, dataloaders, evaluators, hooks, optimizer, AMP wrapper) is
kept identical.

Purpose
-------
A controlled comparison against zod_bevfusion_finetune.py.  The only
difference from finetune is the initialisation strategy: instead of a
single load_from that loads the full LiDAR checkpoint, both modality
branches are warm-started individually from their best unimodal checkpoints.
Schedule, epochs, and all other hyperparameters are identical to finetune,
so any performance difference is attributable solely to the init strategy.

This config is also designed as the MoE-ready dual-init recipe: with both
modality branches already trained, swapping in a MoE fusion block in place
of ConvFuser requires only that module to learn from scratch.

Initialisation — dual-checkpoint selective loading
---------------------------------------------------
Both modality-specific branches are warm-started from their respective
best unimodal checkpoints via ``DualCheckpointInitHook``.

    LiDAR-side  (pts_middle_encoder, pts_backbone, pts_neck, bbox_head)
        ← best_mAP_0.50_epoch_18.pth  from the ZOD LiDAR-only run

    Camera-side (img_backbone, img_neck, view_transform)
        ← best_mAP_0.50_epoch_11.pth  from the ZOD camera-only run

    Fresh init  (fusion_layer only)
        ← random / default initialisation

No top-level ``load_from`` is used.  All initialisation is performed by
``DualCheckpointInitHook`` (see Hook ordering note below).

Module-loading rationale (verified against actual checkpoint keys)
------------------------------------------------------------------
pts_middle_encoder (LiDAR-only ← LiDAR ckpt, 126 keys)
    Core sparse encoder; must come from LiDAR ckpt.

pts_backbone (LiDAR-only ← LiDAR ckpt, 72 keys)
    LiDAR ckpt: blocks.0.0.weight = [128, 256, 3, 3]  → in_channels=256 ✓
    Camera ckpt: blocks.0.0.weight = [128, 80, 3, 3]  → in_channels=80  ✗
    The fusion model also uses in_channels=256 for pts_backbone, so the
    LiDAR checkpoint is shape-compatible and gives a strong warm-start.
    The camera-only checkpoint is architecturally incompatible here.

pts_neck (LiDAR-only ← LiDAR ckpt, 12 keys)
    SECONDFPN taking input channels [128, 256] from pts_backbone.
    Load from LiDAR ckpt to keep pts_backbone + pts_neck consistent
    (they were trained together as a unit).

bbox_head (TransFusionHead ← LiDAR ckpt)
    Detection head loaded from the LiDAR-only checkpoint.  The head
    architecture is identical between the LiDAR-only and fusion configs
    (both take 512-ch input from the neck), so all weights transfer
    without shape mismatch.  Loading from the LiDAR ckpt gives a strong
    detection prior that matches the finetune baseline init.

fusion_layer (ConvFuser) ← fresh random init
    No unimodal counterpart exists; randomly initialised in both this
    config and zod_bevfusion_finetune.py.

pts_voxel_encoder (HardSimpleVFE) — NOT listed
    Confirmed stateless: zero keys in the LiDAR checkpoint.
    HardSimpleVFE has no learned parameters and does not need loading.

view_transform (DepthLSSTransform ← Camera ckpt, 59 keys)
    All tensor shapes verified compatible with the fusion config:
      depthnet.0.weight : [256, 320, 3, 3]  (in=in_ch+64=256+64=320) ✓
      depthnet.6.weight : [258, 256, 1, 1]  (out=D+C=178+80=258) ✓
      frustum           : [178, 88, 156, 3]  (D=(90-1)/0.5=178, fH=88, fW=156) ✓
      dtransform        : 1→8→32→64 conv stack ✓
      downsample        : 80-ch stride-2 conv stack ✓
    All 59 tensors load without any shape mismatch.

Hook ordering note
------------------
``DualCheckpointInitHook`` fires in ``before_run`` with priority=VERY_HIGH,
which is **before** mmengine calls ``resume_or_load()`` (which would process
a ``load_from`` key).  Therefore ``load_from`` is intentionally absent from
this config; setting it would cause ``resume_or_load`` to overwrite the
selective dual-checkpoint init.

Training schedule — matches zod_bevfusion_finetune.py exactly
--------------------------------------------------------------
  lr          = 5e-5
  warmup      = LinearLR(start_factor=1/3, 500 iters)
  cosine      = phase 1 (0→4 ep, eta_min=lr*10) + phase 2 (4→12 ep, eta_min=lr*1e-4)
  momentum    = CosineAnnealingMomentum mirroring the LR phases
  max_epochs  = 12

Changes from zod_bevfusion_finetune.py
---------------------------------------
  1. No ``load_from`` (replaced by DualCheckpointInitHook in custom_hooks).
  2. DualCheckpointInitHook added to custom_hooks loading:
       lidar_ckpt  → pts_middle_encoder, pts_backbone, pts_neck, bbox_head
       camera_ckpt → img_backbone, img_neck, view_transform
"""

_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)

# ---------------------------------------------------------------------------
# No load_from — dual-checkpoint init is performed by DualCheckpointInitHook.
# Setting load_from here would cause mmengine's resume_or_load() to overwrite
# the hook's selective loading.  Leave this commented as documentation.
# ---------------------------------------------------------------------------
# load_from = ...   # intentionally absent; see DualCheckpointInitHook below

# ===== geometry =====
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]
grid_size = [1440, 1440, 40]
sparse_shape = [1440, 1440, 41]
out_size_factor = 8

# ===== dataset =====
class_names = ['pedestrian']
metainfo = dict(classes=class_names, box_type_3d='LiDAR')
dataset_type = 'ZODDataset'
data_root = '/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/'
data_prefix = dict(pts='', CAM_FRONT='')
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# ===== model =====
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[200000, 240000],
            voxelize_reduce=True)),

    # ── camera encoder ──
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        # init_cfg loads ImageNet weights in model.__init__ via init_weights().
        # DualCheckpointInitHook subsequently overwrites img_backbone with the
        # camera-only ZOD checkpoint (stronger than ImageNet alone).
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/tier2/project/p201222/u103958/checkpoints/swin_tiny_patch4_window7_224.pth')),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),

    # ── camera → BEV (adapted to ZOD asymmetric range) ──
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[704, 1248],
        feature_size=[88, 156],
        xbound=[0.0, 108.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 90.0, 0.5],
        downsample=2,
        splat_radius=1,
        aux_depth_loss_weight=0.5),

    # ── fusion (camera BEV 80-ch + LiDAR BEV 256-ch → 256-ch) ──
    # fusion_layer has no unimodal source; starts from random init.
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256),

    # ── LiDAR encoder (identical to zod_lidar_only) ──
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                          (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)),
                          (0, 0)),
        block_type='basicblock'),

    # ── post-fusion BEV backbone / neck ──
    # pts_backbone and pts_neck are loaded from the LiDAR-only checkpoint
    # by DualCheckpointInitHook (see lidar_modules list in custom_hooks).
    # Architecture compatibility verified: LiDAR ckpt pts_backbone uses
    # in_channels=256 — identical to the fusion model.  Camera ckpt uses
    # in_channels=80 and is incompatible.
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    # ── detection head — loaded from LiDAR checkpoint ──
    # bbox_head is loaded from the LiDAR-only checkpoint by DualCheckpointInitHook
    # (see lidar_modules list in custom_hooks).  Architecture is identical between
    # LiDAR-only and fusion configs (both take 512-ch neck output), so all weights
    # transfer without shape mismatch.  This matches the finetune baseline init.
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=500,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=1,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        train_cfg=dict(
            dataset='custom_zod',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(
                    type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            dataset='custom_zod',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type='circle'),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[0.0, -54.0, -10.0, 108.0, 54.0, 10.0],
            score_threshold=0.0,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            code_size=8),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0, alpha=0.25,
            reduction='mean', loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss',
            reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss',
            reduction='mean', loss_weight=0.25)))

# ===== pipelines =====

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
        use_h=True,
        use_w=True,
        max_epoch=10,   # matches finetune config
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
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
        ])
]

# ===== dataloaders =====

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='infos/zod_nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=False,
        use_valid_flag=False,
        with_velocity=False,
        box_type_3d='LiDAR',
        filter_empty_gt=False))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='infos/zod_nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        use_valid_flag=False,
        with_velocity=False,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='infos/zod_nuscenes_infos_test.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        use_valid_flag=False,
        with_velocity=False,
        box_type_3d='LiDAR',
        backend_args=backend_args))

# ===== evaluators =====
val_evaluator = [
    dict(type='IndoorMetric', iou_thr=[0.25, 0.5]),
    dict(type='CenterDistanceMetric', dist_thr=[0.5, 1.0, 2.0, 4.0]),
]
test_evaluator = [
    dict(type='IndoorMetric', iou_thr=[0.25, 0.5]),
    dict(type='CenterDistanceMetric', dist_thr=[0.5, 1.0, 2.0, 4.0]),
]

# ===== visualizer =====
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ===== optimizer / scheduler =====
# Identical to zod_bevfusion_finetune.py for a controlled comparison.
# The short LinearLR warmup stabilises early training while fusion_layer
# (the only randomly-initialised module) is still producing noisy gradients.
# Two-phase cosine + mirrored momentum schedulers match the finetune recipe.
#
# Effective LR profile:
#   iter 0 → 500   : warmup   lr * 0.333 → lr * 1.0
#   epoch 0 → 4    : cosine   lr → eta_min = lr * 10 = 5e-4
#   epoch 4 → 12   : cosine   lr → eta_min = lr * 1e-4 = 5e-9

lr = 5e-5
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=4,
        eta_min=lr * 10,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 1e-4,
        begin=4,
        end=12,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=1,
        begin=4,
        end=12,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale='dynamic')

auto_scale_lr = dict(enable=False)
log_processor = dict(window_size=50)

# ===== hooks =====
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        save_best='mAP_0.50',
        rule='greater'))

# Visualisation epochs matching the 12-epoch finetune run.
_VIS_EPOCHS = (1, 3, 5, 7, 10, 12)

custom_hooks = [
    # ── Dual-checkpoint init (runs before_run, priority=VERY_HIGH) ──────
    # LiDAR-side: pts_middle_encoder (126 keys), pts_backbone (72 keys),
    #   pts_neck (12 keys), bbox_head — all from LiDAR-only checkpoint.
    #   pts_voxel_encoder (HardSimpleVFE) is intentionally NOT listed:
    #   verified stateless (0 keys in checkpoint).
    # Camera-side: img_backbone (187 keys), img_neck (24 keys),
    #   view_transform (59 keys, all shapes verified) — from camera-only ckpt.
    # Fresh init: fusion_layer (ConvFuser) only.
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
            'pts_middle_encoder',   # 126 keys — sparse LiDAR encoder
            'pts_backbone',         # 72 keys  — SECOND, in_channels=256 (compat.)
            'pts_neck',             # 12 keys  — SECONDFPN, trained w/ pts_backbone
            'bbox_head',            # TransFusionHead — identical arch to LiDAR-only
        ],
        camera_modules=[
            'img_backbone',         # 187 keys — Swin-T
            'img_neck',             # 24 keys  — GeneralizedLSSFPN
            'view_transform',       # 59 keys  — DepthLSSTransform (shape-verified)
        ]),
    # ── Visualisation and diagnostic hooks (identical to finetune config) ─
    dict(type='BEVFeatureVisualizationHook',
         vis_epochs=_VIS_EPOCHS),
    dict(type='BEVPredictionVisualizationHook', score_thr=0.15,
         vis_epochs=_VIS_EPOCHS),
    dict(type='BEVValPredictionVisualizationHook', score_thr=0.15,
         vis_epochs=_VIS_EPOCHS),
    dict(type='BEVCameraFeatureVisualizationHook',
         vis_epochs=_VIS_EPOCHS),
    dict(type='DepthTransformDiagnosticHook',
         vis_epochs=_VIS_EPOCHS),
    dict(type='TrainingEfficiencyHook'),
    dict(type='RunSummaryHook'),
    dict(type='DepthProjectionDebugHook', vis_epochs=_VIS_EPOCHS),
    dict(type='ValidationCurveHook',
         metric_keys=('mAP_0.50', 'mAP_0.5m'),
         filename='val_curve_ap_0_50_0_5m'),
]
