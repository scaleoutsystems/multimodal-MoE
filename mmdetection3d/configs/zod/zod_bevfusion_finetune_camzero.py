"""Camera-zeroed ablation config for zod_bevfusion_finetune.

Identical to zod_bevfusion_finetune.py in every respect except that the
model type is changed to BEVFusionCameraZero, which replaces the camera BEV
tensor with zeros immediately before fusion.  Load the same checkpoint as
the full model to measure the LiDAR-only contribution inside the trained
network.
"""
_base_ = ['zod_bevfusion_finetune.py']

model = dict(type='BEVFusionCameraZero')
