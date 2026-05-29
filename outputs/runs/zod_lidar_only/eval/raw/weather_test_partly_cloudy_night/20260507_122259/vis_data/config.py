auto_scale_lr = dict(enable=False)
backend_args = None
class_names = [
    'pedestrian',
]
custom_hooks = [
    dict(type='BEVFeatureVisualizationHook'),
    dict(score_thr=0.15, type='BEVPredictionVisualizationHook'),
    dict(score_thr=0.15, type='BEVValPredictionVisualizationHook'),
    dict(type='TrainingEfficiencyHook'),
    dict(type='RunSummaryHook'),
    dict(
        filename='val_curve_ap_0_50_0_5m',
        metric_keys=(
            'mAP_0.50',
            'mAP_0.5m',
        ),
        type='ValidationCurveHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.BEVFusion.bevfusion',
    ])
data_prefix = dict(pts='')
data_root = '/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/'
dataset_type = 'ZODDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        rule='greater',
        save_best='mAP_0.50',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
grid_size = [
    1440,
    1440,
    40,
]
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'none'
load_from = 'outputs/runs/zod_lidar_only/zod-lidar-only_4543546/best_mAP_0.50_epoch_19.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 5e-05
metainfo = dict(
    box_type_3d='LiDAR', classes=[
        'pedestrian',
    ])
model = dict(
    bbox_head=dict(
        auxiliary=True,
        bbox_coder=dict(
            code_size=8,
            out_size_factor=8,
            pc_range=[
                0.0,
                -54.0,
            ],
            post_center_range=[
                0.0,
                -54.0,
                -10.0,
                108.0,
                54.0,
                10.0,
            ],
            score_threshold=0.0,
            type='TransFusionBBoxCoder',
            voxel_size=[
                0.075,
                0.075,
            ]),
        bn_momentum=0.1,
        common_heads=dict(
            center=[
                2,
                2,
            ], dim=[
                3,
                2,
            ], height=[
                1,
                2,
            ], rot=[
                2,
                2,
            ]),
        decoder_layer=dict(
            cross_attn_cfg=dict(dropout=0.1, embed_dims=128, num_heads=8),
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=128,
                feedforward_channels=256,
                ffn_drop=0.1,
                num_fcs=2),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
            self_attn_cfg=dict(dropout=0.1, embed_dims=128, num_heads=8),
            type='TransformerDecoderLayer'),
        hidden_channel=128,
        in_channels=512,
        loss_bbox=dict(
            loss_weight=0.25, reduction='mean', type='mmdet.L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            reduction='mean',
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_heatmap=dict(
            loss_weight=1.0, reduction='mean', type='mmdet.GaussianFocalLoss'),
        nms_kernel_size=3,
        num_classes=1,
        num_decoder_layers=1,
        num_proposals=500,
        test_cfg=dict(
            dataset='custom_zod',
            grid_size=[
                1440,
                1440,
                40,
            ],
            nms_type='circle',
            out_size_factor=8,
            pc_range=[
                0.0,
                -54.0,
            ],
            voxel_size=[
                0.075,
                0.075,
            ]),
        train_cfg=dict(
            assigner=dict(
                cls_cost=dict(
                    alpha=0.25,
                    gamma=2.0,
                    type='mmdet.FocalLossCost',
                    weight=0.15),
                iou_calculator=dict(coordinate='lidar', type='BboxOverlaps3D'),
                iou_cost=dict(type='IoU3DCost', weight=0.25),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                type='HungarianAssigner3D'),
            code_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dataset='custom_zod',
            gaussian_overlap=0.1,
            grid_size=[
                1440,
                1440,
                40,
            ],
            min_radius=2,
            out_size_factor=8,
            point_cloud_range=[
                0.0,
                -54.0,
                -5.0,
                108.0,
                54.0,
                3.0,
            ],
            pos_weight=-1,
            voxel_size=[
                0.075,
                0.075,
                0.2,
            ]),
        type='TransFusionHead'),
    data_preprocessor=dict(
        pad_size_divisor=32,
        type='Det3DDataPreprocessor',
        voxelize_cfg=dict(
            max_num_points=10,
            max_voxels=[
                200000,
                240000,
            ],
            point_cloud_range=[
                0.0,
                -54.0,
                -5.0,
                108.0,
                54.0,
                3.0,
            ],
            voxel_size=[
                0.075,
                0.075,
                0.2,
            ],
            voxelize_reduce=True)),
    pts_backbone=dict(
        conv_cfg=dict(bias=False, type='Conv2d'),
        in_channels=256,
        layer_nums=[
            5,
            5,
        ],
        layer_strides=[
            1,
            2,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            128,
            256,
        ],
        type='SECOND'),
    pts_middle_encoder=dict(
        block_type='basicblock',
        encoder_channels=(
            (
                16,
                16,
                32,
            ),
            (
                32,
                32,
                64,
            ),
            (
                64,
                64,
                128,
            ),
            (
                128,
                128,
            ),
        ),
        encoder_paddings=(
            (
                0,
                0,
                1,
            ),
            (
                0,
                0,
                1,
            ),
            (
                0,
                0,
                (
                    1,
                    1,
                    0,
                ),
            ),
            (
                0,
                0,
            ),
        ),
        in_channels=4,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        order=(
            'conv',
            'norm',
            'act',
        ),
        sparse_shape=[
            1440,
            1440,
            41,
        ],
        type='BEVFusionSparseEncoder'),
    pts_neck=dict(
        in_channels=[
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            256,
            256,
        ],
        type='SECONDFPN',
        upsample_cfg=dict(bias=False, type='deconv'),
        upsample_strides=[
            1,
            2,
        ],
        use_conv_for_no_stride=True),
    pts_voxel_encoder=dict(num_features=4, type='HardSimpleVFE'),
    type='BEVFusion')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=5e-05, type='AdamW', weight_decay=0.01),
    type='AmpOptimWrapper')
out_size_factor = 8
param_scheduler = [
    dict(
        T_max=8,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=8,
        eta_min=0.0005,
        type='CosineAnnealingLR'),
    dict(
        T_max=12,
        begin=8,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=5e-09,
        type='CosineAnnealingLR'),
    dict(
        T_max=8,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=8,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=12,
        begin=8,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]
point_cloud_range = [
    0.0,
    -54.0,
    -5.0,
    108.0,
    54.0,
    3.0,
]
resume = False
sparse_shape = [
    1440,
    1440,
    41,
]
test_cfg = dict()
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        '/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/infos/zod_nuscenes_infos_weather_test_partly_cloudy_night.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts=''),
        data_root='/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/',
        metainfo=dict(box_type_3d='LiDAR', classes=[
            'pedestrian',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                point_cloud_range=[
                    0.0,
                    -54.0,
                    -5.0,
                    108.0,
                    54.0,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                ],
                meta_keys=[
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'num_pts_feats',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='ZODDataset',
        use_valid_flag=False,
        with_velocity=False),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(iou_thr=[
        0.25,
        0.5,
    ], type='IndoorMetric'),
    dict(dist_thr=[
        0.5,
        1.0,
        2.0,
    ], type='CenterDistanceMetric'),
]
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        point_cloud_range=[
            0.0,
            -54.0,
            -5.0,
            108.0,
            54.0,
            3.0,
        ],
        type='PointsRangeFilter'),
    dict(
        keys=[
            'points',
        ],
        meta_keys=[
            'box_type_3d',
            'sample_idx',
            'lidar_path',
            'num_pts_feats',
        ],
        type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='infos/zod_nuscenes_infos_train.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(pts=''),
        data_root='/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/',
        filter_empty_gt=False,
        metainfo=dict(box_type_3d='LiDAR', classes=[
            'pedestrian',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_attr_label=False,
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    0.9,
                    1.1,
                ],
                translation_std=0.5,
                type='GlobalRotScaleTrans'),
            dict(
                point_cloud_range=[
                    0.0,
                    -54.0,
                    -5.0,
                    108.0,
                    54.0,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    0.0,
                    -54.0,
                    -5.0,
                    108.0,
                    54.0,
                    3.0,
                ],
                type='ObjectRangeFilter'),
            dict(classes=[
                'pedestrian',
            ], type='ObjectNameFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                meta_keys=[
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'transformation_3d_flow',
                    'pcd_rotation',
                    'pcd_scale_factor',
                    'pcd_trans',
                    'lidar_aug_matrix',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='ZODDataset',
        use_valid_flag=False,
        with_velocity=False),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        rot_range=[
            0,
            0,
        ],
        scale_ratio_range=[
            0.9,
            1.1,
        ],
        translation_std=0.5,
        type='GlobalRotScaleTrans'),
    dict(
        point_cloud_range=[
            0.0,
            -54.0,
            -5.0,
            108.0,
            54.0,
            3.0,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            0.0,
            -54.0,
            -5.0,
            108.0,
            54.0,
            3.0,
        ],
        type='ObjectRangeFilter'),
    dict(classes=[
        'pedestrian',
    ], type='ObjectNameFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        meta_keys=[
            'box_type_3d',
            'sample_idx',
            'lidar_path',
            'transformation_3d_flow',
            'pcd_rotation',
            'pcd_scale_factor',
            'pcd_trans',
            'lidar_aug_matrix',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='infos/zod_nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts=''),
        data_root='/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/',
        metainfo=dict(box_type_3d='LiDAR', classes=[
            'pedestrian',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                point_cloud_range=[
                    0.0,
                    -54.0,
                    -5.0,
                    108.0,
                    54.0,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                ],
                meta_keys=[
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'num_pts_feats',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='ZODDataset',
        use_valid_flag=False,
        with_velocity=False),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(iou_thr=[
        0.25,
        0.5,
    ], type='IndoorMetric'),
    dict(dist_thr=[
        0.5,
        1.0,
        2.0,
    ], type='CenterDistanceMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.075,
    0.075,
    0.2,
]
work_dir = '/mnt/tier2/users/u103958/projects/multimodal-MoE/outputs/runs/zod_lidar_only/eval/raw/weather_test_partly_cloudy_night'
