"""
JLC LoRA Model-Only Filtering Helper
------------------------------------

Shared helper for JLC MODEL-only LoRA loader nodes.

Purpose:
    Some LoRA files contain both diffusion-model weights and text-encoder / CLIP
    weights. MODEL-only nodes intentionally do not patch CLIP. Recent ComfyUI
    builds may warn about those text-encoder keys when a MODEL-only loader passes
    the full LoRA state into the normal LoRA loading path.

    This helper removes only keys that are clearly text-encoder / CLIP-side keys
    before MODEL-only application. Ambiguous keys are deliberately preserved so
    ComfyUI can still warn about unexpected missing MODEL-side keys.

Design constraints:
    - Do not suppress all LoRA warnings.
    - Do not mutate cached LoRA state dictionaries.
    - Do not change LoRA strength semantics.
    - Do not change block-weight math or slot behavior.
"""

from __future__ import annotations

from ...jlc_custom_nodes_versions import JLC_LORA_HELPERS_VERSION

import re
from typing import Any, Dict, Tuple


# LoRA-family prefixes used by common Kohya / LyCORIS naming styles.
_LORA_FAMILY_PREFIXES = (
    "lora",
    "loha",
    "lokr",
    "hada",
    "ia3",
    "glora",
)

# Component names that indicate text-encoder / CLIP side LoRA content when they
# appear as the component immediately after the LoRA-family prefix.
_TEXT_ENCODER_COMPONENTS = (
    "te",
    "te1",
    "te2",
    "te3",
    "text_encoder",
    "text_encoder_1",
    "text_encoder_2",
    "text_encoder_3",
    "clip",
    "clip_l",
    "clip_g",
    "clip_bigg",
    "t5",
    "t5xxl",
    "umt5",
)

_TEXT_ENCODER_DIRECT_PREFIXES = (
    "text_encoder.",
    "text_encoder_1.",
    "text_encoder_2.",
    "text_encoder_3.",
    "cond_stage_model.",
    "clip.",
    "clip_l.",
    "clip_g.",
    "clip_bigg.",
    "te.",
    "te1.",
    "te2.",
    "te3.",
    "t5.",
    "t5xxl.",
    "umt5.",
)

# Precompiled conservative matcher for strings such as:
#   lora_te_text_model_encoder_layers_0_...
#   lora_te1_text_model_encoder_layers_0_...
#   loha_te2_text_model_encoder_layers_0_...
#   lora_clip_l_transformer_text_model_...
#   lora_t5xxl_encoder_block_...
# It intentionally does not match lora_unet_*, lora_transformer_*,
# model.diffusion_model.*, diffusion_model.*, or other ambiguous model-side keys.
_FAMILY_TEXT_ENCODER_RE = re.compile(
    r"^(?:"
    + "|".join(re.escape(prefix) for prefix in _LORA_FAMILY_PREFIXES)
    + r")_(?:"
    + "|".join(re.escape(component) for component in _TEXT_ENCODER_COMPONENTS)
    + r")(?=$|[_.])"
)


def is_text_encoder_lora_key(key: str) -> bool:
    """Return True only for keys that are clearly TE/CLIP-side LoRA keys."""

    if not isinstance(key, str):
        return False

    normalized = key.strip().lower().replace("\\", ".").replace("/", ".")

    if not normalized:
        return False

    if normalized.startswith(_TEXT_ENCODER_DIRECT_PREFIXES):
        return True

    if _FAMILY_TEXT_ENCODER_RE.match(normalized):
        return True

    # Diffusers-style nested names sometimes keep a leading module prefix before
    # text_encoder/text_encoder_2. Treat those as explicitly TE-side, while still
    # avoiding broad contains-based matching for words like "encoder" alone.
    if ".text_encoder." in normalized:
        return True
    if ".text_encoder_1." in normalized:
        return True
    if ".text_encoder_2." in normalized:
        return True
    if ".text_encoder_3." in normalized:
        return True

    return False


def filter_lora_state_for_model_only(
    lora_state: Dict[str, Any]
) -> Tuple[Dict[str, Any], int]:
    """
    Return a MODEL-only LoRA state dict and the number of ignored TE/CLIP keys.

    The original dictionary is returned unchanged when no keys are filtered.
    When filtering is needed, a shallow copy is returned so cached LoRA states are
    not modified.
    """

    if not isinstance(lora_state, dict):
        return lora_state, 0

    ignored_keys = [key for key in lora_state if is_text_encoder_lora_key(key)]

    if not ignored_keys:
        return lora_state, 0

    ignored = set(ignored_keys)
    filtered_state = {
        key: value for key, value in lora_state.items() if key not in ignored
    }
    return filtered_state, len(ignored_keys)


def print_model_only_lora_filter_summary(ignored_key_count: int) -> None:
    """Print one compact summary for intentionally ignored TE/CLIP keys."""

    if ignored_key_count > 0:
        print(
            "[JLC LoRA Model Only] Ignored "
            f"{ignored_key_count} text-encoder/CLIP LoRA keys by design."
        )
