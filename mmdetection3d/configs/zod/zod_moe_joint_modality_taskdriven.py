"""Joint-Modality MoE — task-driven routing variant (no context supervision).

Identical to ``zod_moe_joint_modality.py`` (75-epoch converge-to-plateau
schedule, early stopping, GroupNorm joint experts) in every respect
*except* the routing supervision, mirroring the LiDAR-only MoE TD delta:

    * context_aux_cfg : weighted-CE on road_type  →  None
    * gate_input_detach : True  →  False

Everything else — experts, auxiliary MoE losses, optimiser, schedule,
hooks, data pipelines, dual-init — is inherited verbatim.  The unused
``joint_modality_moe.context_head`` paramwise key inherited from the base
matches no parameter and is silently ignored.  The per-context routing
visualisations become no-ops (no ``_ctx_target_field`` on the block).
"""
_base_ = ['./zod_moe_joint_modality.py']

model = dict(
    joint_modality_moe_cfg=dict(
        context_aux_cfg=None,
        gate_input_detach=False,
    ),
)
