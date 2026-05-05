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
    # 512 channels — BEVMoEBlock sits after SECONDFPN (post-neck,
    # pre-bbox_head) and receives the concatenated 512-ch FPN output.
    channels=512,
    num_experts=num_experts,
    k=3,
    num_convs=2,
    # Shazeer importance loss.  Restored to 0.002 (run 4542759 post-mortem).
    # In run 4542759 (TopkGate, γ=0.005) E0 collapsed to top-1=100% by end
    # of epoch 0: the per-expert task-loss signal from γ=0.005 perturbations
    # (~0.5% of feature scale) was too small relative to the balance gradient
    # at 5e-4, so the gate took the lazy one-expert solution immediately.
    # With γ raised to 0.05 (10× stronger per-expert signal) and importance
    # back to 2e-3, the restraining force is now better matched to the per-
    # expert task signal.
    importance_coef=0.002,
    # Shazeer Gaussian-CDF load loss.  No-op for deterministic TopkGate.
    load_coef=0.0,
    # Fedus Switch balance loss.  Raised 0.002 → 0.01 (run 4542759
    # post-mortem).  At 2e-3 the loss was too weak to prevent the
    # winner-take-all collapse of TopkGate early in training.  1e-2
    # matches run 4540532's value, but without the NoisyTopkGate noise
    # inflating f_e artificially — the clean deterministic f_e now
    # reflects true routing imbalance and the gradient bites harder.
    switch_balance_coef=0.01,
    # ST-MoE router z-loss.  Penalises squared log-partition of clean logits,
    # preventing a single expert from dominating without distorting logit rank.
    # Raised 5e-5 → 1e-3 (run 4543230 post-mortem).  In 4543230 clean_logits_std
    # grew unboundedly across epochs (0.59 → 0.73 → 1.08 over ep1-3), causing
    # the dispatch softmax to sharpen and the second/third top-k experts to
    # progressively lose gradient share.  At 5e-5 the per-logit z-loss
    # gradient was ~10⁻⁵, while the task gradient through the gate was ~10⁻³,
    # so z-loss was 100× too weak to anchor the logit scale.  1e-3 puts z-loss
    # on the same order as the task gradient — caps clean_logits_std around
    # ~1.0 and keeps top-k dispatch weights at roughly [0.5, 0.3, 0.2] for
    # k=3 instead of collapsing to a single dominant expert.
    z_loss_coef=1e-3,
    # Residual-delta dispatch gain.  Small-random gamma init (see bev_experts.py)
    # means block(x) ≈ ε·delta at init; residual_gain=1.0 leaves the dispatch
    # weight unscaled once experts diverge.
    residual_gain=1.0,
    # Context-supervised routing.  loss_coef raised 0.10 → 0.20 (Fix #3).
    # With filter_empty_gt=False (Fix #2) the context head now sees balanced
    # road_type labels including highway/rural — doubling the loss weight
    # pushes the z_ctx descriptor to encode more discriminative context
    # structure for the gate to read via z_ctx.detach().  label_smoothing=0.05
    # prevents saturation; class_weights='inverse_frequency' corrects for the
    # remaining ~5× city dominance in the training split even after Fix #2.
    context_aux_cfg=dict(
        target_field='road_type',
        loss_coef=0.20,
        loss_type='weighted_ce',
        label_smoothing=0.05,
        class_weights='inverse_frequency',
    ),
    # Deterministic TopkGate (Fix #1, run 4540532 analysis).
    # Replaces NoisyTopkGate whose learned noise_std head grew to ~2.6 by
    # epoch 1, making the noise/clean_logits_std ratio 1.4–1.8 throughout
    # training.  The training-time top-k was effectively random, the clean
    # deterministic router used at val received no useful gradient, and every
    # val epoch the dominant expert flipped because the clean logit spread
    # was only 0.13–0.40 (small enough that per-update drift changed the
    # top-1 winner).
    # TopkGate with temperature=2.0 (run 4542759 post-mortem).
    # At T=1 with logit spread of 2+ (typical after ep0), the top-2
    # dispatch weights are ~[0.88, 0.12] — the second expert gets only
    # 12% weight, so its task-loss gradient is 7× smaller than the
    # top-1 expert's, accelerating winner-take-all collapse.
    # At T=2 the weights become ~[0.73, 0.27], keeping the w₁·w₂ term
    # in the gate's specialisation gradient (∂L/∂v₁ = w₁·w₂·⟨...⟩)
    # at 0.20 vs 0.11 at T=1 — nearly 2× more gradient reaching the
    # gate from the same logit spread.  Temperature does not change
    # which experts are selected (top-k on raw logits), only how the
    # dispatch weights are mixed.
    gate_type='topk',
    gate_cfg=dict(temperature=2.0),
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
    # Limited spatial augmentation:
    #   - mild scale jitter (±10%)
    #   - rotation disabled to reduce train/val covariate shift on the
    #     routing distribution (rot_range=[0, 0])
    #   - small isotropic translation (std=0.5 m)
    # Re-enable rotation once routing is stable on val.
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
        # Fix #2 (run 4540532 analysis): NuScenesDataset default is
        # filter_empty_gt=True, which drops every frame with no ground-truth
        # pedestrian.  In the ZOD training split this removed ~98% of highway
        # and ~88% of arterial-rural samples, so those road_type classes were
        # effectively absent from training (0.18% highway observed vs 11.13%
        # in the pkl; 1.15% arterial-rural vs 9.90%).  The context auxiliary
        # head never saw highway frames and predicted highway=0 across all 8
        # val epochs of run 4540532.  Setting False retains all frames:
        # pedestrian-free scenes still contribute a meaningful gradient via
        # heatmap/focal "no-object" supervision and ctx_aux_loss, and the
        # router now sees the full scene-type distribution matching val.
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
# Dual-summary routing design:
#
#   router_summary   — feeds the gate (task-driven routing descriptor)
#   context_summary  — feeds context_head (auxiliary context classifier)
#
# Routing-path parameters (gate + router_summary) must learn meaningful
# logit margins for top-k dispatch. Heavy weight decay on these components
# shrinks logit magnitudes and harms expert specialisation, so we use a
# very small decay (decay_mult=0.01 → effective wd ≈ 1e-4) — just enough
# to prevent unbounded logit growth without flattening the gate.
#
# Context-path parameters (context_summary + context_head) form a pure
# classifier trained with weighted CE.  decay_mult=0.05 (effective wd
# ≈ 5e-4) — half the default.  Run 4540062 (post BN→GN fix) showed the
# context head completely collapsed at val: ctx_pred_hist=[0,0,2500,0,0]
# (predicted "city" for every sample), ctx_aux_acc=0.51 (≈ city base
# rate).  With deterministic GN descriptors and decay_mult=0.1 there's
# no opposing force keeping the descriptor encoder discriminative —
# weight decay shrinks the encoder's weights, the descriptor degenerates
# to near-constant, and the head's best constant prediction is the
# majority class.  Lighter decay lets the encoder + head retain enough
# capacity to fit per-sample road_type while the auxiliary CE provides
# the input-dependent signal.
#
    #   bev_moe.gate              — base LR; covers the TopkGate linear layer
    #                                (substring match on 'bev_moe.gate').
    #                                decay_mult=0.01 (effective wd ≈ 1e-4).
#
#   bev_moe.router_summary    — 2× LR; routing descriptor backbone used by
#                                the gate.  decay_mult=0.01.
#
#   bev_moe.context_summary   — 2× LR; context feature extractor for
#                                auxiliary classification.  decay_mult=0.05
#                                (effective wd ≈ 5e-4).
#
#   bev_moe.context_head      — 2× LR; auxiliary context classifier trained
#                                with weighted CE on road_type.
#                                decay_mult=0.05.
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
        # router_summary lr_mult lowered 2.0 → 1.0 (run 4543230 post-mortem).
        # At 2.0 the router input encoder updated twice as fast as the rest of
        # the model, so the gate's input descriptor distribution was a moving
        # target — every epoch the gate had to relearn what the descriptors
        # mean, driving the routing oscillation observed in 4543230 (highway
        # primary went E0 → E4 → E4, arterial-urban went E2 → E2/E3 → E0/E2,
        # E3 collapsed from 24.6% to 7.4% top-1 in one epoch).  At 1.0 the
        # router descriptor evolves at the same pace as the experts and
        # detection head, so the gate has time to settle on stable
        # assignments.  bev_moe.gate stays at 1.0 (default).
        'bev_moe.gate': dict(lr_mult=1.0, decay_mult=0.01),
        'bev_moe.router_summary': dict(lr_mult=1.0, decay_mult=0.01),

        # CONTEXT (lighter decay — see header comment.  GN-based
        # descriptors lack the batch-stats stochasticity that previously
        # kept the encoder discriminative under heavy decay, so the
        # encoder + head need extra capacity to avoid collapsing to
        # majority-class predictions.)
        # lr_mult lowered 2.0 → 1.0 (run 4543230 post-mortem).  The gate
        # input is z_gate = cat([z_router, z_ctx.detach()]) — even though
        # z_ctx is stop-gradient for the gate, its values shift fast at
        # lr_mult=2.0 as the context branch learns road type, presenting
        # the gate with a non-stationary input distribution.  Combined
        # with router_summary at lr_mult=1.0, both halves of z_gate now
        # evolve at the same pace as the rest of the model, eliminating
        # the moving-target effect that drove the routing oscillation
        # observed in 4543230 (clean_logits_std growing 0.59 → 1.08 over
        # ep1-3 with shifting per-road-type assignments).  Mild slow-down
        # of ctx_aux_acc growth is acceptable — the 0.20 loss_coef
        # already provides substantial supervision.
        'bev_moe.context_summary': dict(lr_mult=1.0, decay_mult=0.05),
        'bev_moe.context_head':    dict(lr_mult=1.0, decay_mult=0.05),
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
