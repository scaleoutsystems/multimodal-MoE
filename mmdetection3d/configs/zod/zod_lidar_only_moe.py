_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)

# ---------------------------------------------------------------------------
# Variant D — LiDAR-only MoE
# ---------------------------------------------------------------------------
# Architecture:
#   lidar_bev (256 ch) → BEVMoEBlock → pts_backbone → pts_neck → bbox_head
#
# No camera branch, no ConvFuser, no fusion of any kind.
# BEVMoEBlock routes each sample to one of 6 residual-conv experts based on
# a spatial-aware BEVSummaryHead descriptor + road_type context embedding.
#
# Experts use num_convs=2 to give comparable capacity to variants that flow
# through ConvFuser (which itself is a single conv layer).
#
# Pretrained weights: NuScenes LiDAR-only BEVFusion checkpoint.
# Same transfer semantics as zod_lidar_only.py — MoE-specific modules
# (gate, summary head, context encoder, experts) are randomly initialised
# since they don't exist in the checkpoint.
# ---------------------------------------------------------------------------

load_from = '/mnt/tier2/project/p201222/u103958/checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth'

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]
grid_size = [1440, 1440, 40]
sparse_shape = [1440, 1440, 41]
out_size_factor = 8

class_names = ['pedestrian']
metainfo = dict(classes=class_names, box_type_3d='LiDAR')
dataset_type = 'ZODDataset'
data_root = '/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/'
data_prefix = dict(pts='')
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# ── MoE configuration ────────────────────────────────────────────────────
num_experts = 6

bev_moe_cfg = dict(
    type='BEVMoEBlock',
    channels=256,
    num_experts=num_experts,
    k=2,
    num_convs=2,
    # Switch balance loss coefficient (α).  Reduced from 0.005 → 0.001 to
    # allow mild probability peaking during bootstrap.  At perfect uniform
    # routing L_balance ≡ α, so the floor is now 0.001 instead of 0.005.
    switch_coef=0.005,
    load_coef=0.005,
    # Residual-delta dispatch gain.  Set to num_experts so that at init
    # (Switch-style weight ≈ 1/E) the effective expert contribution
    # g·w ≈ 1 — experts contribute at full scale from step 1, unblocking
    # the detection gradient to the gate.  See BEVMoEBlock class docstring
    # for the full bootstrap rationale and stability caveats.
    residual_gain=float(num_experts),
    router_pool_size=2,
    router_hidden_dim=128,
    router_out_dim=64,
    context_cfg=dict(
        fields=['road_type'],
        embed_dim=16,
        out_dim=64,
    ),
)

# ── Model ─────────────────────────────────────────────────────────────────
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[200000, 240000],
            voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'),
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
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
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
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)),
    # Variant D: LiDAR-only MoE — no fusion_layer, no camera branch.
    bev_moe_cfg=bev_moe_cfg,
)

# ── Pipelines ─────────────────────────────────────────────────────────────
# 'context' must be in meta_keys so that ContextEncoder can read road_type
# from batch_input_metas during routing.

train_pipeline = [
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
        type='GlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[0, 0],
        translation_std=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path',
            'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix',
            'context',
        ])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points'],
        meta_keys=['box_type_3d', 'sample_idx', 'lidar_path',
                   'num_pts_feats', 'context'])
]

# ── Dataloaders ───────────────────────────────────────────────────────────
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

# ── Evaluation ────────────────────────────────────────────────────────────
val_evaluator = [
    dict(type='IndoorMetric', iou_thr=[0.25, 0.5]),
    dict(type='CenterDistanceMetric', dist_thr=[0.5, 1.0, 2.0]),
]
test_evaluator = [
    dict(type='IndoorMetric', iou_thr=[0.25, 0.5]),
    dict(type='CenterDistanceMetric', dist_thr=[0.5, 1.0, 2.0]),
]

# ── Visualizer ────────────────────────────────────────────────────────────
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ── Schedule ──────────────────────────────────────────────────────────────
lr = 5e-5
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=8, eta_min=lr * 10,
        begin=0, end=8, by_epoch=True, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12, eta_min=lr * 1e-4,
        begin=8, end=20, by_epoch=True, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8, eta_min=0.85 / 0.95,
        begin=0, end=8, by_epoch=True, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12, eta_min=1,
        begin=8, end=20, by_epoch=True, convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ── Optimizer ─────────────────────────────────────────────────────────────
# Routing-path param groups (gate / summary head / context encoder) are
# small MLPs with few parameters that need to learn meaningful logit
# margins for top-k dispatch to specialise.  Empirically the dense softmax
# stays near-uniform (margins ~0.05 in logit space) while hard top-1 is
# skewed, suggesting the routing path is over-damped by the same weight
# decay (0.01) and base LR (5e-5) used for the pretrained LiDAR backbone.
#
# We therefore give routing-path parameters their own multipliers via
# MMEngine's paramwise_cfg.custom_keys (matched as substrings against
# parameter names; longest match wins, so these keys cannot leak onto
# bev_moe.experts.*):
#
#   bev_moe.gate.*             — TopkGate.gate linear layer
#   bev_moe.summary.*          — BEVSummaryHead MLP + final LayerNorm
#   bev_moe.context_encoder.*  — ContextEncoder embeddings + proj MLP
#
# lr_mult=3.0   → routing LR 1.5e-4 (vs backbone 5e-5); lets logit
#                 margins grow faster so top-1 choices become confident.
# decay_mult=0.0 → no weight decay on routing params.  Decay on the gate
#                 Linear actively shrinks logit magnitudes, which is
#                 exactly what's preventing margin growth.  Embeddings
#                 and LayerNorm params also standardly get no decay.
#
# Expert CNN blocks (bev_moe.experts.*) are NOT listed here and therefore
# fall through to the default AdamW settings (lr=5e-5, weight_decay=0.01),
# matching the LiDAR backbone / neck / bbox_head.  Architecture, routing
# semantics (k=2, Switch-style weights), residual_gain, and aux losses are
# all unchanged.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale='dynamic',
    paramwise_cfg=dict(
        custom_keys={
            'bev_moe.gate':            dict(lr_mult=3.0, decay_mult=0.0),
            'bev_moe.summary':         dict(lr_mult=3.0, decay_mult=0.0),
            'bev_moe.context_encoder': dict(lr_mult=3.0, decay_mult=0.0),
        },
    ),
)

auto_scale_lr = dict(enable=False)
log_processor = dict(window_size=50)

# Sparse top-k routing means only the selected expert(s) run each forward
# pass, leaving idle experts with no gradient.  Tell DDP to tolerate unused
# parameters rather than crashing on the second iteration.
model_wrapper_cfg = dict(find_unused_parameters=True)

# ── Hooks ─────────────────────────────────────────────────────────────────
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='mAP_0.50',
        rule='greater'))

custom_hooks = [
    dict(type='BEVFeatureVisualizationHook'),
    dict(type='BEVPredictionVisualizationHook', score_thr=0.15),
    dict(type='BEVValPredictionVisualizationHook', score_thr=0.15),
    dict(type='TrainingEfficiencyHook'),
    dict(type='RunSummaryHook'),
    dict(type='ValidationCurveHook',
         metric_keys=('mAP_0.50', 'mAP_0.5m'),
         filename='val_curve_ap_0_50_0_5m'),
    # ── MoE routing hooks ────────────────────────────────────────────
    dict(
        type='MoERoutingHook',
        num_experts=num_experts,
        enable_hook_a=True,
        enable_hook_b=True,
        enable_hook_c=False,
        ap_metric_key='mAP_0.5m',
    ),
    dict(type='ContextRoutingStatsHook'),
    dict(type='ContextExpertUsageVisualizationHook'),
]
