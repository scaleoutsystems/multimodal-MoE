# FOR ZOD_MOE Thesis (ablation study):
# Continue training the best ZOD LiDAR-only checkpoint (epoch 18, best mAP_0.50).


_base_ = ['zod_lidar_only.py']

# ---------------------------------------------------------------------------
# Continue from best ZOD LiDAR-only checkpoint (epoch 18, best mAP_0.50).
# Loads weights only — optimizer/scheduler state is reset.
# Trains for 12 further epochs with a fresh cosine schedule.
# ---------------------------------------------------------------------------
load_from = (
    '/home/users/u103958/projects/multimodal-MoE/outputs/runs/'
    'zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth'
)

train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)

# 12-epoch cosine schedule: short warmup (0→4) then long decay (4→12).
# Keeps the same base lr (5e-5) and warmup peak / decay floor ratios as
# the original 20-epoch schedule, scaled to the shorter horizon.
lr = 5e-5
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=4, eta_min=lr * 10,
        begin=0, end=4, by_epoch=True, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=8, eta_min=lr * 1e-4,
        begin=4, end=12, by_epoch=True, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=4, eta_min=0.85 / 0.95,
        begin=0, end=4, by_epoch=True, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8, eta_min=1,
        begin=4, end=12, by_epoch=True, convert_to_iter_based=True),
]
