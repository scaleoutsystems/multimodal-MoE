_base_ = ['./zod_lidar_only.py']

# =========================================================================
# Overfit sanity test: 20 frames, NO augmentation, CONSTANT LR
#
# Purpose: verify that the box format, coordinate conventions, and target
# generation are correct by overfitting on a tiny subset.  If matched_ious
# climb above ~0.1 within 50-100 epochs, the pipeline is sound.
# =========================================================================

# --- train pipeline: deterministic only, no geometric augmentation -------
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='PointsRangeFilter',
         point_cloud_range=[0.0, -54.0, -5.0, 108.0, 54.0, 3.0]),
    dict(type='ObjectRangeFilter',
         point_cloud_range=[0.0, -54.0, -5.0, 108.0, 54.0, 3.0]),
    dict(type='ObjectNameFilter', classes=['pedestrian']),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path',
        ])
]

# --- tiny training set: 20 frames with GT instances ---------------------
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        ann_file='infos/zod_nuscenes_infos_train_overfit20.pkl',
        pipeline=train_pipeline))

# --- completely disable validation so it never runs ---------------------
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None

# --- constant LR for clean overfit signal --------------------------------
param_scheduler = [
    dict(type='ConstantLR', factor=1.0, begin=0, end=100, by_epoch=True)
]

# --- training schedule: enough epochs to overfit, frequent logging -------
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=9999)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=50))

# --- viz hooks: score_thr=0.2 for clean plots (suppresses noisy clusters) --
custom_hooks = [
    dict(type='BEVFeatureVisualizationHook'),
    dict(type='BEVPredictionVisualizationHook', score_thr=0.2),
    dict(type='BEVValPredictionVisualizationHook', score_thr=0.2),
]
