_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)

# ---------------------------------------------------------------------------
# Variant D — LiDAR-only MoE (context-supervised routing, post-SECONDFPN)
# ---------------------------------------------------------------------------
# Architecture:
#   lidar_bev (256 ch)
#   → pts_backbone (SECOND)          # [128-ch, 256-ch multi-scale]
#   → pts_neck (SECONDFPN)           # 512-ch fused BEV
#   → BEVMoEBlock                    # INSERT HERE (post-neck, pre-head)
#   → bbox_head (TransFusionHead)
#
# No camera branch, no ConvFuser, no fusion of any kind.
# BEVMoEBlock routes each sample to one of `num_experts` residual-conv
# experts based on DUAL BEVResSummaryEncoder descriptors:
#   - router_summary (z_router, 256-d): task/routing branch, shaped by
#     detection + MoE losses.  stem + 3 residual blocks + global avg pool.
#   - context_summary (z_ctx, 256-d): context branch, shaped by weighted CE.
#     Same architecture, separate weights.
# Gate input = cat([z_router, z_ctx.detach()], dim=1) → 512-d.
# Context CE gradient only flows through z_ctx, keeping z_router clean.
#
# Insertion point rationale: post-SECONDFPN features are semantically rich
# (multi-scale fused) and match the ResNet-feature-vector pattern used in
# the reference CIFAR MoE project.  Pre-backbone features are low-level
# geometry; operating there forces the router to make decisions before any
# semantic processing has occurred.
#
# Experts use num_convs=2 and operate on 512-ch input/output to match the
# SECONDFPN output dimensionality expected by TransFusionHead.
#
# Pretrained weights: NuScenes LiDAR-only BEVFusion checkpoint.
# MoE-specific modules (gate, summary encoders, context head, experts) are
# randomly initialised since they don't exist in the checkpoint.
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
# 5 experts.  Note: num_experts no longer needs to match the number of
# context classes — under context-supervised routing the context loss
# only shapes the BEV summary descriptor; expert dispatch stays
# task-driven over the top-k.  Five experts give 5C2 = 10 possible
# top-2 pairs which is enough combinatorial richness for the router
# to learn primary + secondary specialisations.
num_experts = 5

bev_moe_cfg = dict(
    type='BEVMoEBlock',
    # 512 channels — BEVMoEBlock now sits after SECONDFPN (post-neck,
    # pre-bbox_head) and receives the concatenated 512-ch FPN output.
    # Previously 256 (sparse encoder output, pre-backbone).
    channels=512,
    num_experts=num_experts,
    k=2,
    num_convs=2,
    # Shazeer importance loss.  0.002 allows the clean softmax to develop
    # differentiated logit gaps; previous runs showed that higher values
    # collapsed dense_mean_prob to near-uniform, eliminating specialisation.
    importance_coef=0.002,
    # Shazeer Gaussian-CDF load loss.  Active for NoisyTopkGate — balances
    # the noisy exploration dispatch and provides gradient when
    # importance_loss is zeroed out at eval.
    load_coef=0.005,
    # Fedus Switch balance loss.  Computed from clean_topk_idx (deterministic
    # validation-time routing) rather than the noisy training dispatch, so it
    # disciplines the router that is actually evaluated.  Reduced from 0.10 —
    # at the previous coefficient the loss dominated balance pressure and
    # over-regularised the clean router (train routing was near-perfectly
    # uniform at the cost of meaningful specialisation).  0.02 keeps a gentle
    # balance constraint without forcing flat dispatch.
    switch_balance_coef=0.02,
    # ST-MoE router z-loss.  Penalises squared log-partition of clean logits,
    # preventing a single expert from dominating without distorting logit rank.
    z_loss_coef=5e-5,
    # Residual-delta dispatch gain.  g=1 applies the expert delta at full
    # scale.  Drop to 0.5 if grad_norm sustains above 50.
    residual_gain=1.0,
    # Summary encoders: BEVResSummaryEncoder (stem + 3 residual blocks +
    # global avg pool → 256-d descriptor).  Both router_summary and
    # context_summary use these params; no separate config needed.
    # Gate input dim = 256 + 256 = 512.
    #
    # Context-supervised routing.  ``target_field`` selects which ZOD
    # categorical field provides labels for the auxiliary context head.
    # ``loss_type='weighted_ce'`` with ``class_weights='inverse_frequency'``
    # uses built-in inverse-frequency weights for road_type to avoid
    # collapsing to the majority class ('city').  Label smoothing 0.05
    # adds mild over-confidence regularisation.
    context_aux_cfg=dict(
        target_field='road_type',
        loss_coef=0.05,
        loss_type='weighted_ce',
        label_smoothing=0.05,
        class_weights='inverse_frequency',
    ),
    # Noisy top-k gate.  noise_scale=0.25 keeps noise_to_clean_std_ratio
    # well below 1 so the clean (deterministic) router governs training
    # dispatch rather than noise.  temperature=1.5 softens the clean
    # softmax at val, reducing winner-take-all routing from a single
    # dominant expert.  noise_epsilon=1e-3 keeps the softplus noise std
    # well-defined at near-zero late in training.
    gate_type='noisy_topk',
    gate_cfg=dict(
        noise_epsilon=1e-3,
        noise_scale=0.08,
        temperature=1.5,
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
# 'context' must be in meta_keys so that BEVMoEBlock can read the
# configured target_field (road_type) from batch_input_metas to build
# the integer labels consumed by the auxiliary context CE loss.

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
        scale_ratio_range=[0.9, 1.1], #prev 1.0, 1.0
        rot_range=[0, 0], # prev 0, 0
        translation_std=0.5), # prev 0.0
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
# Dual-summary routing design:
#
#   router_summary   — feeds the gate (task-driven routing descriptor)
#   context_summary  — feeds context_head (auxiliary context classifier)
#
# Routing-path parameters (gate + router_summary) must learn meaningful
# logit margins for top-k dispatch. Weight decay on these components
# shrinks logit magnitudes and harms expert specialisation, so
# decay_mult=0.0 is used for all routing parameters. LayerNorm parameters
# also conventionally receive no decay.
#
# Context-path parameters (context_summary + context_head) form a pure
# classifier trained with weighted CE. This branch is prone to overfitting
# (high train accuracy vs lower val accuracy), so we apply light weight
# decay to improve generalisation without affecting routing dynamics.
#
#   bev_moe.gate              — base LR; covers both w_gate and w_noise of
#                                NoisyTopkGate (substring match). Shared
#                                with the Shazeer noise head so the noise
#                                path receives matched updates. No decay.
#
#   bev_moe.router_summary    — 2× LR; routing descriptor backbone used by
#                                the gate. No decay to preserve logit scale.
#
#   bev_moe.context_summary   — 2× LR; context feature extractor for
#                                auxiliary classification. Light decay to
#                                reduce overfitting (effective wd ≈ 1e-4).
#
#   bev_moe.context_head      — 2× LR; auxiliary context classifier trained
#                                with weighted CE on road_type. Light decay
#                                improves generalisation.
#
# Expert CNN blocks (bev_moe.experts.*) are NOT listed here and therefore
# fall through to the default AdamW settings (lr=5e-5, weight_decay=0.01).
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale='dynamic',
    paramwise_cfg=dict(
    custom_keys={
        # ROUTER (no decay)
        'bev_moe.gate': dict(lr_mult=1.0, decay_mult=0.01),
        'bev_moe.router_summary': dict(lr_mult=2.0, decay_mult=0.01),

        # CONTEXT (regularized)
        'bev_moe.context_summary': dict(lr_mult=2.0, decay_mult=0.1),
        'bev_moe.context_head': dict(lr_mult=2.0, decay_mult=0.1),
    },
    )
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
    # ExpertRespawnHook must run *before* MoERoutingHook so that the
    # routing plot for epoch N+1 reflects the post-respawn state.
    # Hook ordering is deterministic on priority: ExpertRespawnHook
    # uses 'NORMAL' (50), MoERoutingHook uses 'BELOW_NORMAL' (60) — so
    # respawn fires first on the after_train_epoch event.
    #
    # dead_threshold_ratio=0.1 means "dead if dispatch < 10% of uniform
    # share": for E=5 that's 0.02 absolute — well above noise level
    # but well below a live expert's steady-state ~0.20.  Every run
    # we've inspected had dead experts clearly below 0.01, so this
    # threshold catches them without false positives.
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
        enable_hook_c=False,
        ap_metric_key='mAP_0.5m',
    ),
    dict(type='ContextRoutingStatsHook'),
    dict(type='ContextExpertUsageVisualizationHook'),
]
