"""
JLC LoRA Loader - Multi Model / Shared Block Weight + CLIP
----------------------------------------------------------

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
    - Dynamic MODEL+CLIP LoRA loader with one shared MODEL block-weight
      vector applied to every active LoRA slot.

    - The node predeclares up to 10 LoRA slots, while the frontend JavaScript
      hides or shows rows based on slot_count.

    - Each active slot provides:
        • LoRA selector
        • MODEL strength
        • CLIP/text-encoder strength

    - One shared block_vector controls MODEL/UNet block weighting for all
      active slots.

- Execution Model
    - slot_count is authoritative in the backend.
    - Slots above slot_count are ignored even if they contain saved values.
    - Hidden slot values remain serialized in the workflow.
    - Active LoRAs are applied sequentially in visible slot order.
    - The shared block_vector affects MODEL/UNet patch weights only.
    - CLIP/text-encoder patches use ordinary per-slot CLIP strength.

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
    "name": "JLC LoRA Loader - Multi Model / Shared Block Weight + CLIP",
    "version": JLC_LORA_LOADER_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Dynamic MODEL+CLIP shared-block-weight LoRA loader for ComfyUI. "
        "Predeclares up to 10 LoRA slots and uses frontend JavaScript to show "
        "or hide rows according to slot_count. The backend treats slot_count "
        "as authoritative, ignores hidden slots, preserves serialized hidden "
        "values, and applies active LoRAs sequentially in visible slot order. "
        "One shared block_vector controls MODEL/UNet block weighting for all "
        "active slots, while CLIP/text-encoder patches use ordinary per-slot "
        "CLIP strength."
    ),
}


from .jlc_lora_dynamic_core import (
    DEFAULT_BLOCK_VECTOR,
    MAX_LORA_SLOTS,
    LoraStateCacheMixin,
    apply_lora_model_block_vector_and_clip,
    block_vector_widget,
    build_model_clip_strength_slots,
    collect_model_clip_strength_slots,
    normalize_slot_count,
    parse_vector_csv,
    print_slot_summary,
    slot_count_widget,
    slot_is_inactive,
)


class JLC_DynamicLoraLoaderSharedBlockWeightModelClip(LoraStateCacheMixin):
    """
    JLC LoRA Loader - Multi Model / Shared Block Weight + CLIP.

    Frontend behavior:
        - All ten LoRA selector + strength_model + strength_clip widgets exist
          in the workflow payload.
        - The JS extension hides slot rows above slot_count.
        - Hidden selector/strength values are retained for later reuse.

    Backend behavior:
        - slot_count is clamped to 1..10 and treated as authoritative.
        - Only slots 1..slot_count are inspected/applied.
        - One block_vector applies to MODEL patches for all visible active LoRAs.
        - Each visible LoRA slot uses its own strength_model_XX and
          strength_clip_XX values.
        - "None" LoRAs are skipped.
        - Slots where strength_model_XX and strength_clip_XX are both 0.0 are
          skipped.
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
            "block_vector": block_vector_widget(
                DEFAULT_BLOCK_VECTOR,
                multiline=False,
                tooltip=(
                    "Shared numeric CSV MODEL block-weight vector applied to "
                    "the MODEL side of every active visible LoRA. CLIP is not "
                    "block-weighted."
                ),
            ),
            **build_model_clip_strength_slots(
                cls.MAX_LORAS,
                strength_model_default=0.0,
                strength_clip_default=0.0,
            ),
        }

        return {"required": required}

    def load_loras(
        self,
        model,
        clip,
        slot_count=1,
        block_vector=DEFAULT_BLOCK_VECTOR,
        **kwargs,
    ):
        count = normalize_slot_count(slot_count, self.MAX_LORAS)
        slots = collect_model_clip_strength_slots(kwargs, count)

        needs_model_block_vector = any(
            not slot_is_inactive(
                slot["lora_name"],
                float(slot["strength_model"]),
                float(slot["strength_clip"]),
            )
            and float(slot["strength_model"]) != 0.0
            for slot in slots
        )
        vector = parse_vector_csv(block_vector) if needs_model_block_vector else [1.0]

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
            model, clip = apply_lora_model_block_vector_and_clip(
                model,
                clip,
                lora_state,
                strength_model,
                strength_clip,
                vector,
            )

        print_slot_summary(
            "JLC-Dynamic-Shared-Block-LoRA-ModelClip",
            count=count,
            active=active,
            inactive=inactive,
            max_slots=self.MAX_LORAS,
        )

        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "JLC_DynamicLoraLoaderSharedBlockWeightModelClip": (
        JLC_DynamicLoraLoaderSharedBlockWeightModelClip
    ),
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_DynamicLoraLoaderSharedBlockWeightModelClip": (
        "\u2003JLC LoRA Loader - Multi Model / Shared Block Weight + CLIP"
    ),
}
