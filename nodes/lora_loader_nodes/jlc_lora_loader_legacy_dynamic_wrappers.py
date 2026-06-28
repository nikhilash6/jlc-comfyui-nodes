"""
JLC Dynamic LoRA Loader Legacy Wrappers
---------------------------------------

- JLC ComfyUI Nodes Collection
  - This module is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for
    image generation pipelines, particularly:
        • Flux-based workflows
        • LoRA experimentation
        • advanced inpainting / outpainting pipelines

- Module Purpose
    - Compatibility wrappers for selected legacy LoRA loader node class_types.

    - These wrappers preserve old workflow loading for the minimal legacy set
      while delegating execution to the new dynamic LoRA loader core.

    - Intended retained legacy nodes:
        • fixed 10-slot LoRA stack
        • fixed 2-slot block-weight LoRA stack

- Compatibility Position
    - New workflows should use the dynamic LoRA loader nodes directly.
    - These wrappers are provided only to reduce breakage for existing saved
      workflows.
    - The old family of ad-hoc fixed-count stack variants is intentionally not
      preserved here.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the LoRA loading architecture in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from ...jlc_custom_nodes_versions import JLC_LORA_LOADER_VERSION

MANIFEST = {
    "name": "JLC Dynamic LoRA Loader Legacy Wrappers",
    "version": JLC_LORA_LOADER_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Minimal compatibility wrapper module for selected legacy JLC LoRA "
        "loader class_types. Preserves old workflow loading for the fixed "
        "10-slot LoRA stack and fixed 2-slot block-weight LoRA stack while "
        "delegating execution to the new dynamic LoRA loader core. New "
        "workflows should use the dynamic LoRA loader nodes directly."
    ),
}


from .jlc_lora_dynamic_core import (
    DEFAULT_BLOCK_VECTOR,
    LoraStateCacheMixin,
    apply_lora_model_block_vector_and_clip,
    apply_lora_model_clip,
    block_vector_widget,
    build_plain_model_only_slots,
    collect_plain_model_only_slots,
    float_strength_widget,
    lora_choices,
    parse_vector_csv,
    print_slot_summary,
    slot_is_inactive,
)


MANIFEST = {
    "name": "JLC LoRA Loader Minimal Legacy Dynamic Wrappers",
    "version": (1, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "Minimal compatibility wrappers for only the fixed 10-stack and fixed "
        "2-LoRA block-weight legacy MODEL+CLIP loader class_type names. "
        "Delegates behavior to jlc_lora_dynamic_core.py."
    ),
}

# -----------------------------------------------------------------------------
# Deprecation Tooltip Definition
# -----------------------------------------------------------------------------
LEGACY_DEPRECATION_TOOLTIP = (
    "Legacy compatibility node. This node is preserved for older saved "
    "workflows, but it will be deprecated in a future JLC node release. "
    "For new or updated workflows, replace it with the new dynamic JLC LoRA "
    "Loader nodes under the LoRA loader family."
)


# -----------------------------------------------------------------------------
# Legacy fixed 10-stack: same strength value applied to MODEL and CLIP.
# -----------------------------------------------------------------------------
class JLC_LoraLoaderTenStack(LoraStateCacheMixin):
    """Legacy fixed 10-slot MODEL+CLIP LoRA stack wrapper."""

    DESCRIPTION = LEGACY_DEPRECATION_TOOLTIP

    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")

    NUMBER_OF_LORAS = 10

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            **build_plain_model_only_slots(cls.NUMBER_OF_LORAS, strength_default=0.0),
        }
        return {"required": required}

    def load_lora(self, model, clip, **kwargs):
        slots = collect_plain_model_only_slots(kwargs, self.NUMBER_OF_LORAS)

        active = []
        inactive = []

        for slot in slots:
            slot_i = slot["i"]
            lora_name = slot["lora_name"]
            strength = float(slot["strength"])

            if slot_is_inactive(lora_name, strength):
                inactive.append(slot_i)
                continue

            active.append(slot_i)
            lora_state = self._load_lora_state(lora_name)
            model, clip = apply_lora_model_clip(
                model,
                clip,
                lora_state,
                strength,
                strength,
            )

        print_slot_summary(
            "JLC-Legacy-LoRA-Stack-10x",
            count=self.NUMBER_OF_LORAS,
            active=active,
            inactive=inactive,
            max_slots=self.NUMBER_OF_LORAS,
        )

        return (model, clip)


# -----------------------------------------------------------------------------
# Legacy 2-LoRA shared block-weight wrapper.
# -----------------------------------------------------------------------------
def _legacy_block_weight_two_slots():
    choices = lora_choices()
    slots = {}
    for i in range(1, 3):
        suffix = f"{i:02d}"
        slots[f"lora_{suffix}"] = (choices,)
        slots[f"strength_{suffix}"] = float_strength_widget(
            0.0,
            tooltip=(
                "Legacy combined strength. Used for both MODEL and CLIP unless "
                "strength_model_XX or strength_clip_XX is set."
            ),
        )
        slots[f"strength_model_{suffix}"] = float_strength_widget(
            0.0,
            tooltip="Optional compatibility override for MODEL strength.",
        )
        slots[f"strength_clip_{suffix}"] = float_strength_widget(
            0.0,
            tooltip="Optional compatibility override for CLIP/text-encoder strength.",
        )
    return slots


def _prefer_override(kwargs, key, fallback):
    try:
        value = float(kwargs.get(key, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return value if value != 0.0 else float(fallback)


class JLC_LoraLoaderBlockWeightTwo(LoraStateCacheMixin):
    """Legacy fixed 2-LoRA shared MODEL block-weight + CLIP wrapper."""

    DESCRIPTION = LEGACY_DEPRECATION_TOOLTIP

    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")

    NUMBER_OF_LORAS = 2

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "block_vector": block_vector_widget(
                DEFAULT_BLOCK_VECTOR,
                multiline=False,
                tooltip="Shared numeric CSV MODEL block-weight vector.",
            ),
            **_legacy_block_weight_two_slots(),
        }
        return {"required": required}

    def load_lora(self, model, clip, block_vector=DEFAULT_BLOCK_VECTOR, **kwargs):
        raw_slots = collect_plain_model_only_slots(kwargs, self.NUMBER_OF_LORAS)

        needs_model_block_vector = False
        normalized_slots = []
        for slot in raw_slots:
            suffix = f"{slot['i']:02d}"
            combined_strength = float(slot["strength"])
            strength_model = _prefer_override(
                kwargs,
                f"strength_model_{suffix}",
                combined_strength,
            )
            strength_clip = _prefer_override(
                kwargs,
                f"strength_clip_{suffix}",
                combined_strength,
            )
            normalized = {
                "i": slot["i"],
                "lora_name": slot["lora_name"],
                "strength_model": strength_model,
                "strength_clip": strength_clip,
            }
            normalized_slots.append(normalized)
            if not slot_is_inactive(slot["lora_name"], strength_model, strength_clip):
                needs_model_block_vector = needs_model_block_vector or strength_model != 0.0

        vector = parse_vector_csv(block_vector) if needs_model_block_vector else [1.0]

        active = []
        inactive = []

        for slot in normalized_slots:
            slot_i = slot["i"]
            lora_name = slot["lora_name"]
            strength_model = float(slot["strength_model"])
            strength_clip = float(slot["strength_clip"])

            if slot_is_inactive(lora_name, strength_model, strength_clip):
                inactive.append(slot_i)
                continue

            active.append(slot_i)
            lora_state = self._load_lora_state(lora_name)
            model, clip = apply_lora_model_block_vector_and_clip(
                model,
                clip,
                lora_state,
                strength_model,
                strength_clip,
                vector,
            )

        print_slot_summary(
            "JLC-Legacy-Block-Weight-Two",
            count=self.NUMBER_OF_LORAS,
            active=active,
            inactive=inactive,
            max_slots=self.NUMBER_OF_LORAS,
        )

        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "JLC_LoraLoaderTenStack": JLC_LoraLoaderTenStack,
    "JLC_LoraLoaderBlockWeightTwo": JLC_LoraLoaderBlockWeightTwo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_LoraLoaderTenStack": "\u2003JLC - LoRA Loader Stack (10) [Legacy]",
    "JLC_LoraLoaderBlockWeightTwo": "\u2003JLC 2-LoRA Loader - Block Weight [Legacy]",
}
