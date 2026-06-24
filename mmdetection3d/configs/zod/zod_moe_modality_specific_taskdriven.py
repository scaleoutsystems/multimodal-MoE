"""Modality-Specific MoE — task-driven routing variant (no context sup.).

Identical to ``zod_moe_modality_specific.py`` (75-epoch converge-to-plateau
schedule, early stopping, symmetric output-space experts, group balance)
in every respect *except* the routing supervision, mirroring the
LiDAR-only MoE TD delta:

    * context_aux_cfg : weighted-CE on road_type  →  None
    * gate_input_detach : True  →  False

``group_balance_coef`` (0.004) is RETAINED — it is independent of context
supervision and governs modality-group routing balance, so removing it
would change more than just the descriptor supervision.  Everything else
— experts, auxiliary MoE losses, optimiser, schedule, hooks, data
pipelines, dual-init — is inherited verbatim.  The unused
``modality_specific_moe.context_head`` paramwise key inherited from the
base matches no parameter and is silently ignored.  The per-context
routing visualisations become no-ops (no ``_ctx_target_field``); the
camera/LiDAR group-mass tracking (enable_hook_c=True) still works.
"""
_base_ = ['./zod_moe_modality_specific.py']

model = dict(
    modality_specific_moe_cfg=dict(
        context_aux_cfg=None,
        gate_input_detach=False,
    ),
)
