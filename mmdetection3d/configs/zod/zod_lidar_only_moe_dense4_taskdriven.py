_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)

# ---------------------------------------------------------------------------
# Variant D-T — LiDAR-only MoE, **task-driven routing** (post-neck)
# ---------------------------------------------------------------------------
# This config is a sibling to ``zod_lidar_only_moe_dense4.py`` (the
# context-supervised dense-4 run).  Architecture, dataset, optimiser,
# scheduler and pipelines are identical; the only difference is how the
# routing path is trained:
#
#   dense4 (context-supervised):
#     z_ctx  = context_summary(x_bev)          # (B, 256)
#     z_gate = z_ctx.detach()                  # stop-grad
#     gate(z_gate) → expert dispatch           # task gradient stops
#                                              # at the detach.
#     ctx_loss = CE(context_head(z_ctx),       # auxiliary CE supervises
#                   road_type_label)           # context_summary directly.
#
#   dense4_taskdriven (this config):
#     z_ctx  = context_summary(x_bev)          # (B, 256)
#     z_gate = z_ctx                           # NO detach — full grad.
#     gate(z_gate) → expert dispatch           # detection loss flows
#                                              # back through softmax →
#                                              # experts → bbox_head and
#                                              # all the way into
#                                              # context_summary.
#     (no context_head, no auxiliary CE — context_aux_cfg=None.)
#
# Why this variant?
#   The dense-4 context-supervised run (4562168) showed that ~8% of the
#   loss budget was being spent on a CE objective whose target
#   (road_type) is at best a proxy for what the gate actually needs to
#   condition on for pedestrian detection.  Road type is observable at
#   data-prep time, but the *useful* routing variable is more likely
#   something detection-task-specific (point density, occlusion, ego
#   speed, etc.) that the gate cannot read off road_type alone.  The
#   task-driven variant lets the encoder + gate co-discover whatever
#   feature minimises detection loss, with no hand-specified prior.
#
#   Side benefit: the BEVResSummaryEncoder (~2.6M params after
#   ``_SUMMARY_POOL_SIZE=2``) is no longer pinned to a 5-way road-type
#   classifier.  All of its capacity is spent shaping a router-friendly
#   descriptor.
#
# What stays the same as dense4:
#   - 4 experts, dense softmax dispatch (k = num_experts), every expert
#     always runs.
#   - importance_coef=0.005 / load_coef=0.0 / switch_balance_coef=0.0 /
#     z_loss_coef=5e-4 — same auxiliary losses on the gate's softmax to
#     prevent collapse onto a single expert and to anchor logit scale.
#   - residual_gain=1.0, num_convs=2, post-neck placement (512-ch BEV).
#   - ctx_gate_warmup_epochs=0 (no temperature schedule).
#   - All paramwise lr_mult=1.0, clip_grad max_norm=10, AdamW lr=5e-5
#     with the same two-phase cosine schedule and 20-epoch budget.
#   - Same train/val/test pipelines and dataloaders (context still in
#     meta_keys so the post-hoc routing-by-road_type analysers can
#     bucket gate decisions for diagnostic plots).
#   - Same find_unused_parameters=True (defensive; under task-driven
#     dense dispatch every parameter actually receives gradient — no
#     branch is conditional — but we keep it on for parity with the
#     other MoE configs).
#
# What changes vs dense4:
#   - context_aux_cfg=None             (no auxiliary CE head)
#   - gate_input_detach=False          (gate sees z_ctx, not z_ctx.detach())
#   - paramwise_cfg drops the
#     'bev_moe.context_head' key       (no such submodule exists when
#                                       context_aux_cfg=None).
#   - The 'bev_moe.context_summary' decay_mult is kept at 0.05.  Under
#     the context-supervised variant the rationale was "the encoder
#     fits a 5-way classifier; full wd flattens descriptors and the
#     head collapses to majority-class".  Under task-driven training
#     the encoder fits a much higher-dimensional implicit objective
#     (whatever shapes the routing softmax to minimise det loss); we
#     still want the encoder to retain expressive capacity, so light
#     decay still applies.
#
# Companion sbatch:
#   tools/sbatch/meluxina_train_zod_lidar_only_moe_dense4_taskdriven.sbatch
# Designed to run *concurrently* with the context-supervised dense4
# run so the two variants can be compared at iso-wall-clock.
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
# Identical to dense4 except for the two task-driven flags
# (context_aux_cfg / gate_input_detach).  See module-top docstring for
# the full motivation.
num_experts = 4

bev_moe_cfg = dict(
    type='BEVMoEBlock',
    channels=512,
    num_experts=num_experts,
    k=num_experts,
    num_convs=2,
    importance_coef=0.005,
    load_coef=0.0,
    switch_balance_coef=0.0,
    z_loss_coef=5e-4,
    residual_gain=1.0,
    # ── Task-driven routing flags ────────────────────────────────────
    # No auxiliary CE.  context_summary is shaped purely by detection
    # gradient flowing back through the gate's softmax.
    context_aux_cfg=None,
    # gate consumes z_ctx (full grad), not z_ctx.detach().  This is
    # what makes context_summary trainable in the absence of an
    # auxiliary CE.  BEVMoEBlock.__init__ enforces the invariant
    # "gate_input_detach=True ⇒ context_aux_cfg must be set"; the
    # opposite combination (this config) is allowed.
    gate_input_detach=False,
    # ── Routing parameters (unchanged from dense4) ───────────────────
    gate_type='dense',
    gate_cfg=dict(temperature=1.0),
    ctx_gate_warmup_epochs=0,
    ctx_gate_temp_high=1.0,
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
    bev_moe_cfg=bev_moe_cfg,
    bev_moe_position='post_neck',
)

# ── Pipelines ─────────────────────────────────────────────────────────────
# 'context' stays in meta_keys even though no context_head is configured.
# Reason: ContextRoutingStatsHook + ContextExpertUsageVisualizationHook
# are post-hoc analysers — they read road_type out of batch_input_metas
# at val time and bucket gate decisions by it, regardless of whether the
# model actually trains a context CE.  Keeping 'context' in meta_keys
# lets us answer "does task-driven routing self-discover road_type
# clusters?" for free, which is the exact question this variant is
# meant to probe.

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
        box_type_3d='LiDAR',
        # Same filter_empty_gt=False as dense4 so the road-type
        # distribution at train matches val (highway / arterial-rural
        # are mostly pedestrian-free).  Without context CE the
        # road-type distribution still matters, indirectly: the
        # detection-via-gate gradient implicitly learns to specialise
        # by whatever scene attribute predicts pedestrian-ness, and
        # under-representing entire scene types would distort that
        # signal.
        filter_empty_gt=False,
    ))

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
# paramwise_cfg differs from dense4 only by dropping the
# 'bev_moe.context_head' key — context_aux_cfg=None means there is no
# context_head submodule, so the key would never match.
#
# The 'bev_moe.context_summary' decay_mult=0.05 is kept.  Under the
# task-driven objective the encoder is trained by gradient flowing
# back from detection loss through the gate's softmax; this gradient
# is small in magnitude (it is gated by p_e (1 - p_e) in the softmax
# Jacobian), so the encoder needs the same kind of capacity-preserving
# light decay that the context-supervised variant uses.
#
# 'bev_moe.gate' decay_mult=0.01 is also kept: the rationale (heavy wd
# flattens routing logits and kills specialisation) holds independently
# of how the gate's input is supervised.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale='dynamic',
    paramwise_cfg=dict(
        custom_keys={
            'pts_backbone': dict(lr_mult=1.0),
            'pts_neck':     dict(lr_mult=1.0),
            'bbox_head':    dict(lr_mult=1.0),
            'bev_moe.gate':            dict(lr_mult=1.0, decay_mult=0.01),
            'bev_moe.context_summary': dict(lr_mult=1.0, decay_mult=0.05),
            # NB: no 'bev_moe.context_head' entry — head does not exist
            # when context_aux_cfg=None.
        },
    )
)

auto_scale_lr = dict(enable=False)
log_processor = dict(window_size=50)

# Same as dense4: kept defensively.  Under dense + task-driven every
# parameter actually receives gradient on every forward (gate, encoder
# and all experts run unconditionally), so this could be safely set to
# False — left True for parity.
model_wrapper_cfg = dict(find_unused_parameters=True)

# ── Hooks ─────────────────────────────────────────────────────────────────
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='mAP_0.50',
        rule='greater'))

# ContextRoutingStatsHook + ContextExpertUsageVisualizationHook are
# kept enabled even though no context_head is configured.  They are
# post-hoc analysers (see hook docstring): they read road_type from
# batch_input_metas at val time and bucket gate decisions by it,
# producing the same diagnostic plots as the context-supervised
# variant.  This is exactly what we want for the comparison: "did the
# task-driven gate self-discover any road_type structure, or did it
# pick a different specialisation axis altogether?".
custom_hooks = [
    dict(type='BEVFeatureVisualizationHook'),
    dict(type='BEVPredictionVisualizationHook', score_thr=0.15),
    dict(type='BEVValPredictionVisualizationHook', score_thr=0.15),
    dict(type='TrainingEfficiencyHook'),
    dict(type='RunSummaryHook'),
    dict(type='ValidationCurveHook',
         metric_keys=('mAP_0.50', 'mAP_0.5m'),
         filename='val_curve_ap_0_50_0_5m'),
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
