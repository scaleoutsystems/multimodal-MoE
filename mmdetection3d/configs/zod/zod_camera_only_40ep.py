"""Camera-only BEVFusion for ZOD pedestrian detection — 40-epoch run.

Identical to zod_camera_only.py (run 4469392, 12 epochs) except:
  - max_epochs 12 → 40
  - decay phase extended from 4–12 (T_max=8) to 4–40 (T_max=36)
  - momentum schedulers updated to match

Motivation: run 4469392 (12 ep) best epoch was 11 with val AP@0.5m=0.2156.
The model had not converged.  This run extends to 40 epochs to match the
40-epoch lidar-only and MoE baselines for a fair comparison.
"""

_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)

load_from = (
    '/home/users/u103958/projects/multimodal-MoE/outputs/runs/'
    'zod_bevfusion_finetune/bevfusion-finetune_4456392/'
    'best_mAP_0.50_epoch_12.pth'
)

# ===== geometry =====
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]
grid_size = [1440, 1440, 40]
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
    type='CameraOnlyBEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32),
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
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/tier2/project/p201222/u103958/checkpoints/'
                       'swin_tiny_patch4_window7_224.pth')),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
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
    pts_backbone=dict(
        type='SECOND',
        in_channels=80,
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
        max_epoch=10,
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
        box_type_3d='LiDAR'))

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
# 40-epoch schedule: warmup 0→500 iters + controlled phase 0→4 epochs
# unchanged; decay phase extended from 4–12 (T_max=8) to 4–40 (T_max=36).
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
        eta_min=lr * 2,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    # decay phase extended to epoch 40 (T_max = 40 - 4 = 36)
    dict(
        type='CosineAnnealingLR',
        T_max=36,
        eta_min=lr * 1e-5,
        begin=4,
        end=40,
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
        T_max=36,
        eta_min=1,
        begin=4,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
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

_VIS_EPOCHS = (1, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40)

custom_hooks = [
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
