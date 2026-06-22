"""
JLC Dynamic LoRA Loader Core Helpers
------------------------------------

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
    - Shared backend support for the JLC Dynamic LoRA Loader node family.

    - Provides common helpers for:
        • dynamic LoRA slot widgets
        • slot_count normalization
        • MODEL-only LoRA patching
        • MODEL+CLIP LoRA patching
        • per-node LoRA state caching
        • block-weight vector parsing
        • MODEL block-weight application

- Dynamic Loader Design
    - Python nodes predeclare up to MAX_LORA_SLOTS widgets.
    - Frontend JavaScript may hide or show rows, but the backend treats
      slot_count as authoritative.
    - Hidden slot values remain serialized in workflows and are ignored
      unless their slot becomes visible again.
    - MODEL-only loaders never patch CLIP.
    - MODEL+CLIP loaders use independent MODEL and CLIP strengths per slot.
    - Block vectors apply only to MODEL/UNet patches.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the LoRA loading architecture in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import comfy.lora
import comfy.sd
import comfy.utils
import folder_paths


MAX_LORA_SLOTS = 10

# Flux-style default carried forward from the existing JLC shared block-weight
# MODEL-only node. vector[0] is the fallback/base ratio; vector[1:] maps across
# encountered MODEL block indices.
DEFAULT_BLOCK_VECTOR = "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1"


# -----------------------------------------------------------------------------
# Widget builders
# -----------------------------------------------------------------------------
def lora_choices() -> List[str]:
    """Return the LoRA dropdown choices with a stable "None" sentinel first."""
    return ["None"] + folder_paths.get_filename_list("loras")


def slot_count_widget(max_slots: int = MAX_LORA_SLOTS, default: int = 1) -> Tuple[str, Dict[str, Any]]:
    """ComfyUI INT widget for authoritative dynamic slot count."""
    return (
        "INT",
        {
            "default": int(default),
            "min": 1,
            "max": int(max_slots),
            "step": 1,
            "display": "number",
            "tooltip": (
                "Authoritative active LoRA slot count. The frontend may hide "
                "rows above this value; the backend ignores them."
            ),
        },
    )


def float_strength_widget(
    default: float = 0.0,
    *,
    min_value: float = -10.0,
    max_value: float = 10.0,
    step: float = 0.01,
    tooltip: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Reusable FLOAT strength widget."""
    opts: Dict[str, Any] = {
        "default": float(default),
        "min": float(min_value),
        "max": float(max_value),
        "step": float(step),
    }
    if tooltip:
        opts["tooltip"] = tooltip
    return ("FLOAT", opts)


def block_vector_widget(
    default: str = DEFAULT_BLOCK_VECTOR,
    *,
    multiline: bool = False,
    tooltip: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """ComfyUI STRING widget for a numeric CSV MODEL block-weight vector."""
    opts: Dict[str, Any] = {
        "multiline": bool(multiline),
        "default": str(default),
    }
    if tooltip:
        opts["tooltip"] = tooltip
    return ("STRING", opts)


def build_plain_model_only_slots(
    max_slots: int = MAX_LORA_SLOTS,
    *,
    strength_default: float = 0.0,
) -> Dict[str, Any]:
    """
    Build per-slot widgets for a plain MODEL-only LoRA loader.

    Produces:
        lora_01, strength_01, ... lora_NN, strength_NN
    """
    choices = lora_choices()
    slots: Dict[str, Any] = {}

    for i in range(1, int(max_slots) + 1):
        suffix = f"{i:02d}"
        slots[f"lora_{suffix}"] = (choices,)
        slots[f"strength_{suffix}"] = float_strength_widget(
            strength_default,
            tooltip="MODEL LoRA strength for this slot.",
        )

    return slots


def build_model_strength_slots(
    max_slots: int = MAX_LORA_SLOTS,
    *,
    strength_default: float = 0.0,
) -> Dict[str, Any]:
    """
    Build per-slot widgets for MODEL-only block-weight loaders.

    Produces:
        lora_01, strength_model_01, ... lora_NN, strength_model_NN
    """
    choices = lora_choices()
    slots: Dict[str, Any] = {}

    for i in range(1, int(max_slots) + 1):
        suffix = f"{i:02d}"
        slots[f"lora_{suffix}"] = (choices,)
        slots[f"strength_model_{suffix}"] = float_strength_widget(
            strength_default,
            tooltip="MODEL LoRA strength for this slot.",
        )

    return slots


def build_model_clip_strength_slots(
    max_slots: int = MAX_LORA_SLOTS,
    *,
    strength_model_default: float = 0.0,
    strength_clip_default: float = 0.0,
) -> Dict[str, Any]:
    """
    Build per-slot widgets for MODEL+CLIP LoRA loaders.

    Produces:
        lora_01, strength_model_01, strength_clip_01, ...
    """
    choices = lora_choices()
    slots: Dict[str, Any] = {}

    for i in range(1, int(max_slots) + 1):
        suffix = f"{i:02d}"
        slots[f"lora_{suffix}"] = (choices,)
        slots[f"strength_model_{suffix}"] = float_strength_widget(
            strength_model_default,
            tooltip="MODEL LoRA strength for this slot.",
        )
        slots[f"strength_clip_{suffix}"] = float_strength_widget(
            strength_clip_default,
            tooltip="CLIP/text-encoder LoRA strength for this slot.",
        )

    return slots


def build_per_slot_block_vector_model_only_slots(
    max_slots: int = MAX_LORA_SLOTS,
    *,
    strength_default: float = 0.0,
    vector_default: str = DEFAULT_BLOCK_VECTOR,
) -> Dict[str, Any]:
    """
    Build per-slot widgets for MODEL-only per-LoRA block-vector loaders.

    Produces:
        lora_01, strength_model_01, block_vector_01, ...
    """
    choices = lora_choices()
    slots: Dict[str, Any] = {}

    for i in range(1, int(max_slots) + 1):
        suffix = f"{i:02d}"
        slots[f"lora_{suffix}"] = (choices,)
        slots[f"strength_model_{suffix}"] = float_strength_widget(
            strength_default,
            tooltip="MODEL LoRA strength for this slot.",
        )
        slots[f"block_vector_{suffix}"] = block_vector_widget(
            vector_default,
            tooltip="MODEL block-weight vector for this LoRA slot.",
        )

    return slots


def build_per_slot_block_vector_model_clip_slots(
    max_slots: int = MAX_LORA_SLOTS,
    *,
    strength_model_default: float = 0.0,
    strength_clip_default: float = 0.0,
    vector_default: str = DEFAULT_BLOCK_VECTOR,
) -> Dict[str, Any]:
    """
    Build per-slot widgets for MODEL+CLIP per-LoRA block-vector loaders.

    Produces:
        lora_01, strength_model_01, strength_clip_01, block_vector_01, ...
    """
    choices = lora_choices()
    slots: Dict[str, Any] = {}

    for i in range(1, int(max_slots) + 1):
        suffix = f"{i:02d}"
        slots[f"lora_{suffix}"] = (choices,)
        slots[f"strength_model_{suffix}"] = float_strength_widget(
            strength_model_default,
            tooltip="MODEL LoRA strength for this slot.",
        )
        slots[f"strength_clip_{suffix}"] = float_strength_widget(
            strength_clip_default,
            tooltip="CLIP/text-encoder LoRA strength for this slot.",
        )
        slots[f"block_vector_{suffix}"] = block_vector_widget(
            vector_default,
            tooltip="MODEL block-weight vector for this LoRA slot.",
        )

    return slots


# -----------------------------------------------------------------------------
# Slot count and slot collectors
# -----------------------------------------------------------------------------
def normalize_slot_count(slot_count: Any, max_slots: int = MAX_LORA_SLOTS) -> int:
    """Clamp slot_count to the valid 1..max_slots range."""
    try:
        count = int(slot_count)
    except (TypeError, ValueError):
        count = 1

    return max(1, min(int(max_slots), count))


def _float_from_kwargs(kwargs: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(kwargs.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def collect_plain_model_only_slots(kwargs: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    """Collect lora_XX + strength_XX slots."""
    slots: List[Dict[str, Any]] = []

    for i in range(1, int(count) + 1):
        suffix = f"{i:02d}"
        slots.append(
            {
                "i": i,
                "lora_name": kwargs.get(f"lora_{suffix}", "None"),
                "strength": _float_from_kwargs(kwargs, f"strength_{suffix}", 0.0),
            }
        )

    return slots


def collect_model_strength_slots(kwargs: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    """Collect lora_XX + strength_model_XX slots."""
    slots: List[Dict[str, Any]] = []

    for i in range(1, int(count) + 1):
        suffix = f"{i:02d}"
        slots.append(
            {
                "i": i,
                "lora_name": kwargs.get(f"lora_{suffix}", "None"),
                "strength_model": _float_from_kwargs(
                    kwargs,
                    f"strength_model_{suffix}",
                    0.0,
                ),
            }
        )

    return slots


def collect_model_clip_strength_slots(kwargs: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    """Collect lora_XX + strength_model_XX + strength_clip_XX slots."""
    slots: List[Dict[str, Any]] = []

    for i in range(1, int(count) + 1):
        suffix = f"{i:02d}"
        slots.append(
            {
                "i": i,
                "lora_name": kwargs.get(f"lora_{suffix}", "None"),
                "strength_model": _float_from_kwargs(
                    kwargs,
                    f"strength_model_{suffix}",
                    0.0,
                ),
                "strength_clip": _float_from_kwargs(
                    kwargs,
                    f"strength_clip_{suffix}",
                    0.0,
                ),
            }
        )

    return slots


def collect_per_slot_vector_model_only_slots(
    kwargs: Dict[str, Any],
    count: int,
    *,
    default_vector: str = DEFAULT_BLOCK_VECTOR,
) -> List[Dict[str, Any]]:
    """Collect lora_XX + strength_model_XX + block_vector_XX slots."""
    slots: List[Dict[str, Any]] = []

    for i in range(1, int(count) + 1):
        suffix = f"{i:02d}"
        slots.append(
            {
                "i": i,
                "lora_name": kwargs.get(f"lora_{suffix}", "None"),
                "strength_model": _float_from_kwargs(
                    kwargs,
                    f"strength_model_{suffix}",
                    0.0,
                ),
                "block_vector": kwargs.get(f"block_vector_{suffix}", default_vector),
            }
        )

    return slots


def collect_per_slot_vector_model_clip_slots(
    kwargs: Dict[str, Any],
    count: int,
    *,
    default_vector: str = DEFAULT_BLOCK_VECTOR,
) -> List[Dict[str, Any]]:
    """Collect lora_XX + strength_model_XX + strength_clip_XX + block_vector_XX slots."""
    slots: List[Dict[str, Any]] = []

    for i in range(1, int(count) + 1):
        suffix = f"{i:02d}"
        slots.append(
            {
                "i": i,
                "lora_name": kwargs.get(f"lora_{suffix}", "None"),
                "strength_model": _float_from_kwargs(
                    kwargs,
                    f"strength_model_{suffix}",
                    0.0,
                ),
                "strength_clip": _float_from_kwargs(
                    kwargs,
                    f"strength_clip_{suffix}",
                    0.0,
                ),
                "block_vector": kwargs.get(f"block_vector_{suffix}", default_vector),
            }
        )

    return slots


def hidden_slot_numbers(count: int, max_slots: int = MAX_LORA_SLOTS) -> List[str]:
    """Return hidden/ignored slot numbers as strings for logging."""
    return [str(i) for i in range(int(count) + 1, int(max_slots) + 1)]


def slot_is_inactive(lora_name: Any, *strengths: float) -> bool:
    """True if the slot should not apply any LoRA work."""
    if not lora_name or str(lora_name) == "None":
        return True
    if strengths and all(float(s) == 0.0 for s in strengths):
        return True
    return False


def print_slot_summary(
    tag: str,
    *,
    count: int,
    active: Sequence[Any],
    inactive: Sequence[Any],
    max_slots: int = MAX_LORA_SLOTS,
) -> None:
    """Consistent dynamic-node console summary."""
    ignored = hidden_slot_numbers(count, max_slots)
    print(
        f"[{tag}] slot_count={count} | "
        f"Active: {', '.join(map(str, active)) or 'none'} | "
        f"Inactive visible: {', '.join(map(str, inactive)) or 'none'} | "
        f"Ignored hidden: {', '.join(ignored) or 'none'}"
    )


# -----------------------------------------------------------------------------
# LoRA state cache mixin
# -----------------------------------------------------------------------------
class LoraStateCacheMixin:
    """Per-node-instance LoRA state_dict cache."""

    def __init__(self) -> None:
        self._loaded_lora_states: Dict[str, Dict[str, Any]] = {}

    def _load_lora_state(self, lora_name: str) -> Dict[str, Any]:
        lora_path = folder_paths.get_full_path("loras", lora_name)

        if lora_path is None:
            raise ValueError(f"Could not resolve LoRA path for '{lora_name}'")

        if lora_path in self._loaded_lora_states:
            return self._loaded_lora_states[lora_path]

        state = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self._loaded_lora_states[lora_path] = state
        return state


# -----------------------------------------------------------------------------
# Plain LoRA application helpers
# -----------------------------------------------------------------------------
def _key_name(key: Any) -> str:
    return str(key[0] if isinstance(key, tuple) else key)


def _looks_like_clip_or_text_key(key: Any) -> bool:
    name = _key_name(key).lower()
    return (
        "clip" in name
        or "text" in name
        or "encoder" in name
        or "cond_stage" in name
        or "transformer.text" in name
    )


def apply_lora_model_only(model: Any, lora_state: Dict[str, Any], strength_model: float) -> Any:
    """
    Apply LoRA patches to MODEL only.

    This intentionally builds only the UNet/model key map via
    comfy.lora.model_lora_keys_unet(model.model). CLIP keys are not loaded and
    are skipped defensively if any text-encoder-style key appears.
    """
    if float(strength_model) == 0.0:
        return model

    key_map = comfy.lora.model_lora_keys_unet(model.model)
    loaded = comfy.lora.load_lora(lora_state, key_map)

    new_model = model.clone()

    for key, weights in loaded.items():
        if _looks_like_clip_or_text_key(key):
            continue
        new_model.add_patches({key: weights}, float(strength_model))

    return new_model


def apply_lora_model_clip(
    model: Any,
    clip: Any,
    lora_state: Dict[str, Any],
    strength_model: float,
    strength_clip: float,
) -> Tuple[Any, Any]:
    """
    Apply a LoRA to MODEL and CLIP using ComfyUI's standard loader.

    This is the helper for non-block-weight MODEL+CLIP dynamic nodes.
    """
    if float(strength_model) == 0.0 and float(strength_clip) == 0.0:
        return model, clip

    return comfy.sd.load_lora_for_models(
        model,
        clip,
        lora_state,
        float(strength_model),
        float(strength_clip),
    )


def apply_lora_clip_only(
    model: Any,
    clip: Any,
    lora_state: Dict[str, Any],
    strength_clip: float,
) -> Any:
    """
    Apply only CLIP/text-encoder patches and return the new CLIP.

    Used by MODEL-block-weight + CLIP variants, where MODEL is patched manually
    with block ratios and CLIP is patched normally.
    """
    if clip is None or float(strength_clip) == 0.0:
        return clip

    _unused_model, new_clip = comfy.sd.load_lora_for_models(
        model,
        clip,
        lora_state,
        0.0,
        float(strength_clip),
    )
    return new_clip


# -----------------------------------------------------------------------------
# Block-vector parsing and MODEL block-weight helpers
# -----------------------------------------------------------------------------
def parse_vector_csv(csv_text: Any) -> List[float]:
    """
    Numeric-only CSV parser for block-vector widgets.

    Accepts:
        "1,0,0,0,1,1,1"

    Returns:
        list[float]
    """
    if csv_text is None:
        raise ValueError("block_vector is required")

    parts = [p.strip() for p in str(csv_text).strip().split(",") if p.strip() != ""]

    if not parts:
        raise ValueError("block_vector cannot be empty")

    vec: List[float] = []

    for p in parts:
        try:
            vec.append(float(p))
        except ValueError as exc:
            raise ValueError(
                f"block_vector must be numeric CSV only. Bad token: '{p}'"
            ) from exc

    return vec


def _parse_unet_num(two_chars: str) -> int:
    """
    Matches Inspire Pack-style behavior:
        "0." -> 0
        "12" -> 12
    """
    if len(two_chars) >= 2 and two_chars[1] == ".":
        return int(two_chars[0])

    return int(two_chars)


def _strip_diffusion_model_prefix(key: str) -> str:
    if key.startswith("diffusion_model."):
        return key[len("diffusion_model.") :]
    return key


def compute_model_block_weights(
    model: Any,
    lora_state: Dict[str, Any],
    vector: Sequence[float],
) -> List[Tuple[Any, Any, float]]:
    """
    Compute MODEL-only block-weight patch tuples for a LoRA state_dict.

    Semantics:
        vector[0] = base ratio for non-matched / "other" model keys
        vector[1:] = ratios consumed by distinct block index while walking:
            input_blocks
            middle_blocks
            output_blocks
            double_blocks
            single_blocks

    If the vector runs out, the last vector value is reused.
    CLIP/text-encoder LoRA keys are intentionally ignored.
    """
    if not vector:
        raise ValueError("block_vector cannot be empty")

    key_map = comfy.lora.model_lora_keys_unet(model.model)
    loaded = comfy.lora.load_lora(lora_state, key_map)

    input_blocks: List[Tuple[Any, Any, int]] = []
    middle_blocks: List[Tuple[Any, Any, int]] = []
    output_blocks: List[Tuple[Any, Any, int]] = []
    double_blocks: List[Tuple[Any, Any, int]] = []
    single_blocks: List[Tuple[Any, Any, int]] = []
    others: List[Tuple[Any, Any]] = []

    for key, weights in loaded.items():
        if _looks_like_clip_or_text_key(key):
            continue

        key_name = _key_name(key)
        k_unet = _strip_diffusion_model_prefix(key_name)

        if k_unet.startswith("input_blocks."):
            num = k_unet[len("input_blocks.") : len("input_blocks.") + 2]
            input_blocks.append((key, weights, _parse_unet_num(num)))

        elif k_unet.startswith("middle_block."):
            num = k_unet[len("middle_block.") : len("middle_block.") + 2]
            middle_blocks.append((key, weights, _parse_unet_num(num)))

        elif k_unet.startswith("output_blocks."):
            num = k_unet[len("output_blocks.") : len("output_blocks.") + 2]
            output_blocks.append((key, weights, _parse_unet_num(num)))

        elif k_unet.startswith("double_blocks."):
            num = k_unet[len("double_blocks.") : len("double_blocks.") + 2]
            double_blocks.append((key, weights, _parse_unet_num(num)))

        elif k_unet.startswith("single_blocks."):
            num = k_unet[len("single_blocks.") : len("single_blocks.") + 2]
            single_blocks.append((key, weights, _parse_unet_num(num)))

        else:
            others.append((key, weights))

    input_blocks.sort(key=lambda x: x[2])
    middle_blocks.sort(key=lambda x: x[2])
    output_blocks.sort(key=lambda x: x[2])
    double_blocks.sort(key=lambda x: x[2])
    single_blocks.sort(key=lambda x: x[2])

    vector_values = list(vector)
    base_ratio = float(vector_values[0])

    if len(vector_values) == 1:
        # One-value vector means: use same ratio everywhere.
        vector_values = [base_ratio, base_ratio]

    block_weights: List[Tuple[Any, Any, float]] = []

    if base_ratio != 0:
        for key, weights in others:
            block_weights.append((key, weights, base_ratio))

    vector_i = 1
    last_block_num: Optional[int] = None
    current_ratio = vector_values[vector_i] if vector_i < len(vector_values) else vector_values[-1]

    ordered_blocks = (
        input_blocks
        + middle_blocks
        + output_blocks
        + double_blocks
        + single_blocks
    )

    for key, weights, block_num in ordered_blocks:
        if last_block_num != block_num:
            if vector_i < len(vector_values):
                current_ratio = float(vector_values[vector_i])
                vector_i += 1
            else:
                current_ratio = float(vector_values[-1])

        last_block_num = block_num

        if current_ratio != 0:
            block_weights.append((key, weights, current_ratio))

    return block_weights


def apply_model_block_weights(
    model: Any,
    block_weights: Iterable[Tuple[Any, Any, float]],
    strength_model: float,
) -> Any:
    """Apply MODEL block-weight patch tuples to a cloned model."""
    if float(strength_model) == 0.0:
        return model

    new_model = model.clone()

    for key, weights, ratio in block_weights:
        if float(ratio) == 0.0:
            continue
        new_model.add_patches({key: weights}, float(strength_model) * float(ratio))

    return new_model


def apply_lora_model_only_with_block_vector(
    model: Any,
    lora_state: Dict[str, Any],
    strength_model: float,
    vector: Sequence[float],
) -> Any:
    """Apply a MODEL-only LoRA with a MODEL block-weight vector."""
    if float(strength_model) == 0.0:
        return model

    block_weights = compute_model_block_weights(model, lora_state, vector)
    return apply_model_block_weights(model, block_weights, strength_model)


def apply_lora_model_block_vector_and_clip(
    model: Any,
    clip: Any,
    lora_state: Dict[str, Any],
    strength_model: float,
    strength_clip: float,
    vector: Sequence[float],
) -> Tuple[Any, Any]:
    """
    Apply a LoRA with block-weighted MODEL patches and normal CLIP patches.

    MODEL receives strength_model * block_vector_ratio.
    CLIP receives ordinary strength_clip via ComfyUI's standard loader.
    """
    new_model = apply_lora_model_only_with_block_vector(
        model,
        lora_state,
        strength_model,
        vector,
    )
    new_clip = apply_lora_clip_only(model, clip, lora_state, strength_clip)
    return new_model, new_clip
