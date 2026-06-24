"""Modality-Specific MoE — scraped_weather context-supervised variant.

Identical to ``zod_moe_modality_specific.py`` (75-epoch converge-to-plateau
schedule, early stopping, symmetric output-space experts, group balance)
in every respect *except* the context descriptor that supervises the
routing gate:

    road_type (5-way)  →  scraped_weather (9-way)

Only ``model.modality_specific_moe_cfg.context_aux_cfg.target_field`` is
overridden; the rest of the MoE block (including group_balance_coef),
auxiliary losses, optimiser, schedule, hooks, data pipelines, and
dual-init are inherited verbatim.  The 9-way vocab is resolved
automatically from ``ZOD_FIELD_REGISTRY['scraped_weather']`` and the
context routing / usage hooks auto-discover the field from the block's
``_ctx_target_field`` attribute, so they report weather groupings with no
further changes.
"""
_base_ = ['./zod_moe_modality_specific.py']

model = dict(
    modality_specific_moe_cfg=dict(
        context_aux_cfg=dict(target_field='scraped_weather'),
    ),
)
