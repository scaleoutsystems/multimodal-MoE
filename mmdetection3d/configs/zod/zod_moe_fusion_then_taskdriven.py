"""Fusion-then-MoE — task-driven routing variant (no context supervision).

Identical to ``zod_moe_fusion_then.py`` (75-epoch converge-to-plateau
schedule, early stopping, GroupNorm experts, bf16) in every respect
*except* the routing supervision, mirroring the LiDAR-only MoE TD delta
(``zod_lidar_only_moe_dense4_taskdriven_30ep.py`` vs the context-supervised
``zod_lidar_only_moe_dense4_30ep.py``):

    * context_aux_cfg : weighted-CE on road_type  →  None
      (no auxiliary context head; the gate is supervised purely by the
      detection task loss).
    * gate_input_detach : True  →  False
      (task gradients now flow into the gate-input summary so routing can
      adapt directly from the task objective, exactly as in the LiDAR-only
      TD run).

Everything else — experts, auxiliary MoE losses, optimiser, schedule,
hooks, data pipelines, dual-init — is inherited verbatim.  The unused
``bev_moe.context_head`` paramwise key inherited from the base matches no
parameter (there is no context head) and is silently ignored.  The
per-context routing visualisations become no-ops (no ``_ctx_target_field``
on the block), exactly as in the LiDAR-only TD configuration.
"""
_base_ = ['./zod_moe_fusion_then.py']

model = dict(
    bev_moe_cfg=dict(
        context_aux_cfg=None,
        gate_input_detach=False,
    ),
)
