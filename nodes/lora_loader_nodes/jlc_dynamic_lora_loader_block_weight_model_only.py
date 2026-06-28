"""
JLC LoRA Loader - Multi-Model / Block Weight
-------------------------------------------------

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
    - Dynamic MODEL-only LoRA loader with a separate MODEL block-weight
      vector for each active LoRA slot.

    - The node predeclares up to 10 LoRA slots, while the frontend JavaScript
      hides or shows rows based on slot_count.

    - Each active slot provides:
        • LoRA selector
        • MODEL strength
        • per-slot MODEL block_vector

- Execution Model
    - slot_count is authoritative in the backend.
    - Slots above slot_count are ignored even if they contain saved values.
    - Hidden slot values remain serialized in the workflow.
    - Active LoRAs are applied sequentially in visible slot order.
    - Each block_vector affects only the MODEL/UNet patches for its own slot.
    - CLIP is never patched by this node.
    - Text-encoder/CLIP LoRA keys are ignored by design before loading.

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
    "name": "JLC LoRA Loader - Multi-Model / Block Weight",
    "version": JLC_LORA_LOADER_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Dynamic MODEL-only per-slot block-weight LoRA loader for ComfyUI. "
        "Predeclares up to 10 LoRA slots and uses frontend JavaScript to show "
        "or hide rows according to slot_count. The backend treats slot_count "
        "as authoritative, ignores hidden slots, preserves serialized hidden "
        "values, and applies active LoRAs sequentially in visible slot order. "
        "Each active slot has its own MODEL strength and MODEL/UNet "
        "block_vector. CLIP is not patched; text-encoder/CLIP LoRA keys are ignored by design."
    ),
}

from .jlc_lora_model_only_filter import (
    filter_lora_state_for_model_only,
    print_model_only_lora_filter_summary,
)

from .jlc_lora_dynamic_core import (
    DEFAULT_BLOCK_VECTOR,
    LoraStateCacheMixin,
    MAX_LORA_SLOTS,
    apply_lora_model_only_with_block_vector,
    build_per_slot_block_vector_model_only_slots,
    collect_per_slot_vector_model_only_slots,
    normalize_slot_count,
    parse_vector_csv,
    print_slot_summary,
    slot_count_widget,
    slot_is_inactive,
)


class JLC_DynamicLoraLoaderBlockWeightModelOnly(LoraStateCacheMixin):
    """
    JLC LoRA Loader - Multi-Model / Block Weight.

    Frontend behavior:
        - All ten slot widget groups exist in the workflow payload.
        - The JS extension hides slot groups above slot_count.
        - Hidden slot values are retained for later reuse.

    Backend behavior:
        - slot_count is clamped to 1..10 and treated as authoritative.
        - Only slots 1..slot_count are inspected/applied.
        - Each slot has its own LoRA selector, strength_model, and block_vector.
        - "None" LoRAs and strength_model == 0.0 are skipped.
        - Hidden slots are ignored even if they contain nonzero values.
        - MODEL-only block-weight patching is delegated to jlc_lora_dynamic_core.
    """

    FUNCTION = "load_loras"
    CATEGORY = "loaders"

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    MAX_LORAS = MAX_LORA_SLOTS
    NUMBER_OF_LORAS = MAX_LORA_SLOTS

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        required = {
            "model": ("MODEL",),
            "slot_count": slot_count_widget(cls.MAX_LORAS, default=1),
            **build_per_slot_block_vector_model_only_slots(
                cls.MAX_LORAS,
                strength_default=0.0,
                vector_default=DEFAULT_BLOCK_VECTOR,
            ),
        }

        return {"required": required}

    def load_loras(self, model, slot_count=1, **kwargs):
        count = normalize_slot_count(slot_count, self.MAX_LORAS)
        slots = collect_per_slot_vector_model_only_slots(
            kwargs,
            count,
            default_vector=DEFAULT_BLOCK_VECTOR,
        )

        active = []
        inactive = []
        ignored_te_clip_total = 0

        for slot in slots:
            slot_number = slot["i"]
            lora_name = slot["lora_name"]
            strength_model = float(slot["strength_model"])

            if slot_is_inactive(lora_name, strength_model):
                inactive.append(str(slot_number))
                continue

            vector = parse_vector_csv(slot["block_vector"])
            lora_state = self._load_lora_state(lora_name)
            lora_state, ignored_te_clip = filter_lora_state_for_model_only(
                lora_state
            )
            ignored_te_clip_total += ignored_te_clip

            model = apply_lora_model_only_with_block_vector(
                model,
                lora_state,
                strength_model,
                vector,
            )

            active.append(str(slot_number))

        print_model_only_lora_filter_summary(ignored_te_clip_total)

        print_slot_summary(
            "JLC-Dynamic-Block-Weight-LoRA-ModelOnly",
            count=count,
            active=active,
            inactive=inactive,
            max_slots=self.MAX_LORAS,
        )

        return (model,)


NODE_CLASS_MAPPINGS = {
    "JLC_DynamicLoraLoaderBlockWeightModelOnly": (
        JLC_DynamicLoraLoaderBlockWeightModelOnly
    ),
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_DynamicLoraLoaderBlockWeightModelOnly": (
        "\u2003JLC LoRA Loader - Multi-Model / Block Weight"
    ),
}
