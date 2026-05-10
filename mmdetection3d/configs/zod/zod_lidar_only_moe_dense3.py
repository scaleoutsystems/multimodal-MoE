_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)

# ---------------------------------------------------------------------------
# Variant D — LiDAR-only MoE (context-supervised routing, post-neck)
# ---------------------------------------------------------------------------
# Architecture (bev_moe_position='post_neck'):
#   lidar_bev (256 ch from pts_middle_encoder)
#   → pts_backbone (SECOND)          # [128-ch, 256-ch multi-scale]
#   → pts_neck (SECONDFPN)           # 512-ch fused BEV
#   → BEVMoEBlock                    # INSERT HERE (after SECONDFPN)
#   → bbox_head (TransFusionHead)
#
# Rationale for post-neck placement (reverted from pre-backbone after 4551963):
#   Pre-backbone placement (runs 4546931, 4551963) was originally chosen with
#   the hypothesis that the pretrained SECOND + SECONDFPN BN stack would
#   *absorb* the init-time expert perturbation before the head saw it.  In
#   practice the opposite happens: the perturbation is sample-dependent
#   (different routing decisions per sample), so the 13 pretrained BN
#   layers downstream operate far from their running-mean / running-var
#   sweet spot and the perturbation compounds layer-by-layer.  Empirically
#   this produced a 6× longer FP16 init shock than post-neck:
#       post_neck   (4544033): loss_heatmap < 50 by iter ~250
#       pre_backbone (4551963): loss_heatmap < 50 by iter ~1500
#   With identical _LAST_BN_GAMMA_STD=0.05 across all four runs (verified
#   by git history), gamma cannot explain the 6× difference — only
#   position can.  See canvas:
#       canvases/small-gammas-vs-moe-position.canvas.tsx
#   Reverting to post_neck localises the perturbation to the head, which
#   is being fine-tuned anyway and adapts within tens of iterations.
#   Experts now operate on 512-ch SECONDFPN output.
#
# Gate: single BEVResSummaryEncoder (z_ctx, 256-d), dense softmax dispatch.
#   gate_type='dense' replaces top-k dispatch with a full-softmax mixture
#   (every expert always runs, weighted by ``full_softmax_probs``):
#
#       x_out = x + Σ_{e=1..E}  p_e · (expert_e(x) − x)
#
#   Motivation (canvas: lidar-moe-ap-gap-diagnosis): runs 4552697 (E=4,
#   k=2, loss_coef=0.10, z_loss=1e-3) and 4554362 (loss_coef=0.05, z_loss=
#   2e-4) both lost ~10 pp of mAP@0.50 vs the no-MoE baseline (best 0.394–
#   0.409 vs 0.515).  The dense softmax sat within ~10 pp of uniform but
#   top-2 dispatch is a binary include/exclude — tiny logit perturbations
#   flipped which experts received gradient on val, the bbox head saw a
#   different feature ensemble each val pass, and AP oscillated (E1
#   swung 19% → 54% across adjacent epochs).  Three failure modes stem
#   from the cliff (gradient starvation, discontinuous switching, train↔
#   val routing mismatch); dense softmax dispatch resolves all three at
#   the cost of E× expert FLOPs per sample.
#
#   Temperature annealing is disabled (ctx_gate_warmup_epochs=0) — T was a
#   fix for cliff brittleness and is irrelevant under dense.  T=1.0
#   throughout.
#
# No camera branch, no ConvFuser, no fusion of any kind.
# Experts use num_convs=2 and operate on 512-ch input/output to match
# the SECONDFPN output (post_neck placement).
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
# 3 experts, dense softmax dispatch (k = num_experts = 3).
# Corrected version of run 4560094 — same 3-expert capacity, but with
# clip_grad max_norm restored to 10 and eta_min phase-1 restored to 5e-4
# (matching the baseline 4543546), and all lr_mult=1.0 (already correct
# in 4560094 but clipping was limiting second-half gains).
#
# Routing evidence from 4560094 (confirmed at ep8) and 4557330:
#   - ZOD has 3 natural semantic regimes:
#       E0 = rural  (smaller-rural + arterial-rural)
#       E1 = highway (93% top-1 by ep8)
#       E2 = urban   (city + arterial-urban)
#   - 4557330 (4 experts) confirmed: E0 was chronically dead (4–6%
#     top-1 across all 16 epochs).  A 4th expert finds no role.
#   - arterial-urban correctly sits between E0/E2 (~50/50), reflecting
#     its genuinely ambiguous semantic position.
# 3 experts is the right inductive bias for this dataset.
num_experts = 3

bev_moe_cfg = dict(
    type='BEVMoEBlock',
    # 512 channels — BEVMoEBlock sits after pts_neck (SECONDFPN) and
    # receives the 512-ch fused BEV.  Reverted from pre-backbone (256-ch)
    # after run 4551963 post-mortem: pre-backbone forced sample-dependent
    # expert perturbations through 13 pretrained BN layers (10 in SECOND
    # + 3 in SECONDFPN), producing a 6× longer FP16 init shock than
    # post-neck with identical γ.  Post-neck localises the perturbation
    # to TransFusionHead which is being fine-tuned anyway.  See canvas:
    # canvases/small-gammas-vs-moe-position.canvas.tsx
    channels=512,
    num_experts=num_experts,
    # Dense dispatch: k is forced to num_experts internally regardless
    # of the value here, but we set it explicitly to make the regime
    # obvious in the config.
    k=num_experts,
    num_convs=2,
    # Shazeer importance loss — soft balance pressure on the full
    # pre-top-k softmax.  Still bites under dense (penalises uneven
    # sum-of-probs across experts).  Kept at 0.001 (per 4551963 post-
    # mortem) — small enough not to crush nascent specialisation, large
    # enough to prevent collapse onto a single expert.
    importance_coef=0.005,
    # Shazeer Gaussian-CDF load loss — no-op under TopkGate (no noise
    # to integrate over) and irrelevant under dense (every expert is
    # always selected).  Left at 0.
    load_coef=0.0,
    # Fedus Switch balance loss — DROPPED for dense.  With k=E every
    # expert is selected on every sample, so f_e = 1/E uniformly and
    # the loss collapses to the constant α·E·Σ(1/E)·P_e = α with no
    # specialisation gradient.  ``BEVMoEBlock`` short-circuits this
    # term to zero whenever ``gate_type='dense'`` regardless of the
    # coefficient, but we set 0.0 here as well to make intent explicit.
    switch_balance_coef=0.0,
    # ST-MoE router z-loss — anchors the logit scale.  Set to 5e-4
    # (canvas recommendation, between the 1e-3 and 2e-4 settings tried
    # in 4552697 / 4554362).  Under dense the cliff is gone, so the
    # exact logit scale matters less; we just want std anchored in a
    # reasonable range (target ~[0.7, 1.5]) to keep the softmax neither
    # near-uniform (no specialisation) nor saturated (no balance
    # pressure for importance_loss).
    z_loss_coef=5e-4,  # LAST RUN: 1e-3 (4552697), 2e-4 (4554362)
    # Residual-delta dispatch gain.  Small-random gamma init (see bev_experts.py)
    # means block(x) ≈ ε·delta at init; residual_gain=1.0 leaves the dispatch
    # weight unscaled once experts diverge.
    residual_gain=1.0,
    # Context-supervised routing — reduced loss_coef from 0.10 → 0.03.
    # In the 4562173 / 4562168 dense runs ctx CE sat at ~1.1 mid-training,
    # so loss_coef=0.10 was contributing ~0.11 to total loss — about 8%
    # of the total budget shaped by road_type rather than detection (see
    # canvas dense-moe-vs-baseline-4562173-4562168 §5).  Lowering to 0.03
    # cuts that to ~2.5% while still providing enough gradient to keep
    # z_ctx context-discriminative: ctx_aux_acc was 0.71 at val with
    # coef=0.10 and the per-class signal is strong (highway 92%, city
    # 60%), so we have headroom to reduce coef without losing the
    # routing signal.  Combined with _SUMMARY_POOL_SIZE=2 in
    # BEVMoEBlock, total context_summary footprint shrinks from ~4M to
    # ~2.6M params and its share of the optimiser budget shrinks from
    # ~8% to ~2.5% — leaving more capacity for the detection task.
    context_aux_cfg=dict(
        target_field='road_type',
        loss_coef=0.03,
        loss_type='weighted_ce',
        label_smoothing=0.05,
        class_weights='inverse_frequency',
    ),
    # Dense softmax dispatch.  Every expert always runs, weighted by
    # the full pre-top-k softmax (no top-k cliff).  See canvas:
    # canvases/lidar-moe-ap-gap-diagnosis.canvas.tsx for the failure
    # analysis that motivated this switch.
    gate_type='dense',
    gate_cfg=dict(temperature=1.0),
    # Temperature annealing — DISABLED for dense.  The warmup schedule
    # was a brittleness fix for the top-k cliff (high T flattens the
    # softmax, smoothing what would otherwise be sharp top-k flips
    # during early training).  Under dense small logit changes already
    # produce small mixing-weight changes, so the schedule has nothing
    # to fix; T=1.0 throughout.
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
    # Variant D: LiDAR-only MoE — no fusion_layer, no camera branch.
    bev_moe_cfg=bev_moe_cfg,
    # Place MoE after SECONDFPN (post_neck).  Reverted from 'pre_backbone'
    # after the 4551963 post-mortem: routing perturbations cascading
    # through 13 pretrained BN layers caused a 6× longer FP16 init shock
    # than post_neck.  γ (_LAST_BN_GAMMA_STD=0.05) was held constant
    # across all four runs, so position — not γ — drives the shock
    # duration.  See canvas: canvases/small-gammas-vs-moe-position.canvas.tsx
    bev_moe_position='post_neck',
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
# Single-summary routing design:
#
#   context_summary  — feeds the gate via z_ctx.detach() AND the auxiliary
#                      context_head (full grad).
#   gate             — small Linear (256 → num_experts), task-driven via
#                      the dispatch path (top-k weights → expert outputs →
#                      detection loss).
#   experts          — 4 residual conv blocks (bev_moe.experts.*).
#
# Routing-path parameters (the gate Linear) must learn meaningful logit
# margins for top-k dispatch.  Heavy weight decay on the gate shrinks
# logit magnitudes and harms expert specialisation, so we use a very small
# decay (decay_mult=0.01 → effective wd ≈ 1e-4) — just enough to prevent
# unbounded logit growth without flattening the gate.
#
# Context-path parameters (context_summary + context_head) form a pure
# classifier trained with weighted CE.  decay_mult=0.05 (effective wd
# ≈ 5e-4) — half the default.  Run 4540062 (post BN→GN fix) showed the
# context head completely collapsed at val with decay_mult=0.1: weight
# decay shrank the encoder's weights, the descriptor degenerated to
# near-constant, and the head's best constant prediction was the majority
# class ("city").  Lighter decay lets the encoder + head retain enough
# capacity to fit per-sample road_type while the auxiliary CE provides
# the input-dependent signal.
#
# Pretrained-component LR multiplier (canvas: lidar-moe-ap-gap-diagnosis):
# pts_backbone, pts_neck, bbox_head are loaded from the NuScenes lidar-
# only BEVFusion checkpoint and were producing solid mAP_0.50 in the
# baseline lidar-only run (4543546).  Inserting a freshly-initialised
# MoE block post-FPN perturbs the feature distribution that
# TransFusionHead expects; in 4544033 this perturbation pulled the head
# off-distribution (mAP_0.50 dropped 8–12 pp vs baseline) and intermittent
# fp16 overflows eventually crashed the Hungarian assigner at epoch 13.
# Slowing pretrained components to lr_mult=0.1 was the original fix.
#
# Post-mortem of run 4557330 (canvas: moe-run-comparison): bbox_head at
# 0.3× peaked at 1.5e-4 — 3.3× slower than the baseline (4543546) where
# bbox_head gets full LR and reaches mAP_0.50=0.515.  The MoE run was
# already behind the baseline at epoch 5 (0.255 vs 0.294) and never
# caught up.  Meanwhile the gate at 1.0× peaked at 5e-4 while the
# backbone it depends on was only at 5e-5 — the gate over-updated
# relative to the feature distribution it was routing over, causing AP
# oscillation (0.255→0.214→0.250→0.239→0.228 across epochs 5-9).
#
# Fixes relative to 4557330:
#   1. bbox_head/pts_backbone/pts_neck all raised to 1.0× — matches
#      baseline (4543546) treatment.  The 0.1×/0.3× suppression in 4557330
#      was the root cause of underperformance: backbone at 5e-6 effective
#      LR was essentially frozen, forcing the MoE to route over static
#      features.  The gate at 1.0× then over-updated relative to the frozen
#      backbone, causing AP oscillation across epochs 5-9.
#   2. clip_grad max_norm raised 2 → 10 — matches baseline.  max_norm=2
#      was bundled with the LR suppression as a joint stability package for
#      the top-k sparse runs; the dense gate has smooth soft-weighted
#      gradients and does not need it.  Keeping 2 while restoring lr_mult=1.0
#      would have partially cancelled the LR restoration on large-gradient
#      steps.
#   3. Warmup eta_min restored lr*8 → lr*10 (4e-4 → 5e-4) — matches
#      baseline scheduler exactly.
#
#   pts_backbone, pts_neck — lr_mult=1.0.
#   bbox_head              — lr_mult=1.0.
#   bev_moe.gate           — lr_mult=1.0, decay_mult=0.01 (light wd to
#                            prevent logit magnitude collapse).
#   bev_moe.context_summary — lr_mult=1.0, decay_mult=0.05.
#   bev_moe.context_head   — lr_mult=1.0, decay_mult=0.05.
#   bev_moe.experts.*      — falls through to AdamW defaults
#                            (lr_mult=1.0, weight_decay=0.01).
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale='dynamic',
    paramwise_cfg=dict(
        custom_keys={
            # All components at 1.0× LR — NuScenes pretraining is a
            # different sensor/class setup from ZOD, so backbone/neck
            # need substantial adaptation just like the head.  With
            # bev_moe_position='post_neck' the MoE sits after the
            # backbone so there is no routing perturbation flowing back
            # through it; the original 0.1× rationale no longer applies.
            'pts_backbone': dict(lr_mult=1.0),
            'pts_neck':     dict(lr_mult=1.0),
            'bbox_head':    dict(lr_mult=1.0),

            # GATE — lr_mult=1.0 (uniform with all other components).
            # decay_mult=0.01 kept: normal wd (0.01) would shrink gate
            # logit magnitudes and flatten the softmax, killing expert
            # specialisation.  Effective wd = 1e-4.
            'bev_moe.gate':            dict(lr_mult=1.0, decay_mult=0.01),

            # CONTEXT (lighter decay — see header comment.  GN-based
            # descriptors lack the batch-stats stochasticity that
            # previously kept the encoder discriminative under heavy
            # decay, so the encoder + head need extra capacity to avoid
            # collapsing to majority-class predictions.)
            'bev_moe.context_summary': dict(lr_mult=1.0, decay_mult=0.05),
            'bev_moe.context_head':    dict(lr_mult=1.0, decay_mult=0.05),
        },
    )
)

auto_scale_lr = dict(enable=False)
log_processor = dict(window_size=50)

# Under dense dispatch every expert runs on every forward, so all expert
# parameters receive gradient and DDP would not strictly need
# find_unused_parameters=True for that reason alone.  We keep it enabled
# defensively for the BEV summary encoder + context head, whose grad path
# is conditional on ``context_aux_cfg`` being set, and to keep this knob
# uniform across MoE configs (some of which still use sparse top-k).
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
    # ExpertRespawnHook runs before MoERoutingHook so the routing plot
    # for epoch N+1 reflects the post-respawn state.  Priority ordering:
    # ExpertRespawnHook='NORMAL' (50), MoERoutingHook='BELOW_NORMAL' (60).
    #
    # Under dense dispatch the hook is effectively a safety net: every
    # expert always runs and receives gradient, so the lottery-winner
    # death mode that the hook was designed for cannot occur.  An
    # expert can only fall below dead_threshold (0.033 absolute for
    # E=3, i.e. 10% of the uniform share) if the gate has collapsed
    # into a near-degenerate softmax — a strong signal that something
    # is wrong upstream.  Kept enabled with the same parameters as the
    # top-k runs for parity; expected to fire 0 times under dense.
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
