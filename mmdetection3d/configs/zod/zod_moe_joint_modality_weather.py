"""Joint-Modality MoE — scraped_weather context-supervised routing variant.

Identical to ``zod_moe_joint_modality.py`` (75-epoch converge-to-plateau
schedule, early stopping, GroupNorm joint experts) in every respect
*except* the context descriptor that supervises the routing gate:

    road_type (5-way)  →  scraped_weather (9-way)

Only ``model.joint_modality_moe_cfg.context_aux_cfg.target_field`` is
overridden; the rest of the MoE block, auxiliary losses, optimiser,
schedule, hooks, data pipelines, and dual-init are inherited verbatim.
The 9-way vocab is resolved automatically from
``ZOD_FIELD_REGISTRY['scraped_weather']`` and the context routing / usage
hooks auto-discover the field from the block's ``_ctx_target_field``
attribute, so they report weather groupings with no further changes.
"""
_base_ = ['./zod_moe_joint_modality.py']

model = dict(
    joint_modality_moe_cfg=dict(
        context_aux_cfg=dict(target_field='scraped_weather'),
    ),
)
