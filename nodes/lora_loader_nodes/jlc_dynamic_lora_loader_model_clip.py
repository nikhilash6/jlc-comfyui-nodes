"""
JLC LoRA Loader - Multi-Model / CLIP
-------------------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for
    image generation pipelines, particularly:
        • Flux-based workflows
        • LoRA experimentation
        • advanced inpainting / outpainting pipelines

- Node Purpose
    - Dynamic MODEL+CLIP LoRA loader with a user-selected active slot count.

    - The node predeclares up to 10 LoRA slots, while the frontend JavaScript
      hides or shows rows based on slot_count.

    - Each active slot provides:
        • LoRA selector
        • MODEL strength
        • CLIP/text-encoder strength

- Execution Model
    - slot_count is authoritative in the backend.
    - Slots above slot_count are ignored even if they contain saved values.
    - Hidden slot values remain serialized in the workflow.
    - Active LoRAs are applied sequentially in visible slot order.
    - MODEL and CLIP strengths are controlled independently per LoRA slot.

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
    "name": "JLC LoRA Loader - Multi-Model / CLIP",
    "version": JLC_LORA_LOADER_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Dynamic MODEL+CLIP LoRA loader for ComfyUI. Predeclares up to 10 LoRA "
        "slots and uses frontend JavaScript to show or hide rows according to "
        "slot_count. The backend treats slot_count as authoritative, ignores "
        "hidden slots, preserves serialized hidden values, and applies active "
        "LoRAs sequentially in visible slot order. Each slot has independent "
        "MODEL and CLIP/text-encoder strengths."
    ),
}

from .jlc_lora_dynamic_core import (
    MAX_LORA_SLOTS,
    LoraStateCacheMixin,
    apply_lora_model_clip,
    build_model_clip_strength_slots,
    collect_model_clip_strength_slots,
    normalize_slot_count,
    print_slot_summary,
    slot_count_widget,
    slot_is_inactive,
)


class JLC_DynamicLoraLoaderModelClip(LoraStateCacheMixin):
    """
    JLC LoRA Loader - Multi-Model / CLIP.

    Frontend behavior:
        - All ten slot widget groups exist in the workflow payload.
        - The JS extension hides slot groups above slot_count.
        - Hidden selector/strength values are retained for later reuse.

    Backend behavior:
        - slot_count is clamped to 1..10 and treated as authoritative.
        - Only slots 1..slot_count are inspected/applied.
        - Each slot has its own LoRA selector, strength_model, and strength_clip.
        - "None" LoRAs and slots where both strengths are 0.0 are skipped.
        - Hidden slots are ignored even if they contain nonzero values.
        - LoRAs are applied sequentially in visible slot order.
    """

    FUNCTION = "load_loras"
    CATEGORY = "loaders"

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")

    MAX_LORAS = MAX_LORA_SLOTS
    NUMBER_OF_LORAS = MAX_LORA_SLOTS

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "slot_count": slot_count_widget(cls.MAX_LORAS, default=1),
            **build_model_clip_strength_slots(
                cls.MAX_LORAS,
                strength_model_default=0.0,
                strength_clip_default=0.0,
            ),
        }

        return {"required": required}

    def load_loras(self, model, clip, slot_count=1, **kwargs):
        count = normalize_slot_count(slot_count, self.MAX_LORAS)
        slots = collect_model_clip_strength_slots(kwargs, count)

        active = []
        inactive = []

        for slot in slots:
            slot_i = slot["i"]
            lora_name = slot["lora_name"]
            strength_model = float(slot["strength_model"])
            strength_clip = float(slot["strength_clip"])

            if slot_is_inactive(lora_name, strength_model, strength_clip):
                inactive.append(slot_i)
                continue

            active.append(slot_i)

            lora_state = self._load_lora_state(lora_name)
            model, clip = apply_lora_model_clip(
                model,
                clip,
                lora_state,
                strength_model,
                strength_clip,
            )

        print_slot_summary(
            "JLC-Dynamic-LoRA-Loader-ModelClip",
            count=count,
            active=active,
            inactive=inactive,
            max_slots=self.MAX_LORAS,
        )

        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "JLC_DynamicLoraLoaderModelClip": JLC_DynamicLoraLoaderModelClip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_DynamicLoraLoaderModelClip":
        "\u2003JLC LoRA Loader - Multi-Model / CLIP",
}
