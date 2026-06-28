"""
JLC Dynamic ControlNet Orchestrator - Advanced
----------------------------------------------

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
    - The **JLC Dynamic ControlNet Orchestrator - Advanced** is a single
      grow/shrink Advanced ControlNet orchestration node for ComfyUI.

    - It replaces locally generated fixed-slot Advanced variants with one
      dynamic node that predeclares up to ten ControlNet slots and treats
      `slot_count` as the authoritative active-slot limit.

    - The node keeps the Advanced Orchestrator's built-in ControlNet selector
      and loader behavior, while delegating model residency to the shared
      JLC cache core instead of using a local OrderedDict cache.

    - It preserves the same non-recursive execution model:
            • Independent ControlNet instances per slot
            • `.copy()`-based isolation (no shared execution state)
            • Execution on the same latent input
            • Streaming weighted fusion of outputs

    - Composition remains:
            combined = Σ (w_i · C_i(x))

- Dynamic Slot Design
    - Python predeclares MAX_CONTROLNET_SLOTS = 10.
    - `slot_count` is clamped to 1..10 and is authoritative.
    - Slots above `slot_count` remain serialized in workflow JSON but are
      ignored by the backend.
    - Frontend JavaScript may hide/show slot rows and image sockets without
      deleting values or links.
    - Image inputs are optional lazy inputs so hidden or inactive slots should
      not force upstream preprocessors to execute.

- Built-in Loader and Shared Cache
    - ControlNets are selected via dropdowns and loaded internally.
    - Base ControlNet models are cached through `jlc_model_cache_core.py`:
            key = controlnet::<normalized full path>
            family = "controlnet"
    - The default ControlNet family capacity is four resident base models.
    - Cached base models are never used directly for conditioning execution;
      each active slot still uses independent `.copy()` isolation.

- Execution Model
    - Slot selector semantics:
            • "DISABLED" → hard bypass
            • "SHARE_PREVIOUS" → inherit last valid promoted ControlNet
            • named model → load/reuse selected ControlNet from shared cache

    - Promotion occurs ONLY after early bypass validation.

    - Early bypass conditions:
            • Missing image
            • Zero strength
            • Zero weight
            • Invalid (start, end) interval

    → Ensures:
            ✔ No inactive slot can influence downstream execution
            ✔ No inactive slot can promote or alter SHARE_PREVIOUS state
            ✔ No order-dependent contamination via slot reuse

- Critical Correctness Guarantees
            • Slot-order invariance for alpha = 1 where expected
            • Zero cross-contamination via `.copy()`
            • Cache-safe execution with shared base-model residency only
            • Single-ControlNet fallback to native ApplyAdvanced-style semantics
            • Multi-ControlNet path remains non-recursive and additive

- ⚠️ Experimental Code
    - This node represents a non-canonical formulation of ControlNet
      interaction that diverges from ComfyUI’s native chained execution model.
    - Behavior is stable and deterministic, but not guaranteed to reproduce
      all edge-case behaviors of the canonical implementation.
    - Multi-GPU execution is not implemented; current compatibility hooks are best-effort and untested.
    - Intended for advanced workflows and controlled experimentation.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

MANIFEST = {
    "name": "JLC Dynamic ControlNet Orchestrator - Advanced",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Single dynamic Advanced ControlNet Orchestrator for ComfyUI. "
        "Predeclares up to 10 ControlNet slots and treats slot_count as "
        "authoritative while frontend JavaScript hides/shows rows. Uses the "
        "shared JLC model cache core for built-in ControlNet loading with a "
        "default family capacity of four resident base ControlNets. Preserves "
        "the original non-recursive weighted additive fusion algorithm, "
        "copy-based per-slot isolation, SHARE_PREVIOUS semantics, early bypass "
        "promotion rules, and native ApplyAdvanced-style single-ControlNet "
        "fallback behavior. Uses the shared JLC ControlNet composition core; "
        "fusion math is unchanged."
    ),
}


import copy
from typing import Any, Dict, List, Sequence

import folder_paths
import comfy.controlnet

try:
    from .engines.jlc_model_cache_core import (
        cache_info,
        get_or_load_model,
        make_controlnet_cache_key,
        set_controlnet_cache_capacity,
    )
except ImportError:  # Supports direct loose-module imports during local testing.
    from .engines.jlc_model_cache_core import (  # type: ignore
        cache_info,
        get_or_load_model,
        make_controlnet_cache_key,
        set_controlnet_cache_capacity,
    )

try:
    from .jlc_controlnet_composition_core import JLC_ComposedControlNet
except ImportError:  # Supports direct loose-module imports during local testing.
    from jlc_controlnet_composition_core import JLC_ComposedControlNet  # type: ignore


DEBUG = True

MAX_CONTROLNET_SLOTS = 10
DEFAULT_CONTROLNET_CACHE_SIZE = 4

# -----------------------------------------------------------------------------
# Shared JLC ControlNet Cache
# -----------------------------------------------------------------------------
set_controlnet_cache_capacity(DEFAULT_CONTROLNET_CACHE_SIZE)


def get_controlnet(path: str):
    """Load/reuse a base ControlNet through the shared JLC model cache core."""

    key = make_controlnet_cache_key(path)

    return get_or_load_model(
        key,
        lambda: comfy.controlnet.load_controlnet(path),
        family="controlnet",
        model_path=path,
        role="controlnet",
        policy="lru_family_capacity",
        max_loaded_for_family=DEFAULT_CONTROLNET_CACHE_SIZE,
        metadata={"loader": "comfy.controlnet.load_controlnet"},
    )

# ------------------------------------------------------------
# 🧠 Shared Core Fusion Wrapper
# ------------------------------------------------------------
# JLC_ComposedControlNet is imported from jlc_controlnet_composition_core.py.
# This keeps the Advanced Orchestrator and standalone Composition node on the
# same conservative non-recursive fusion implementation.


# -----------------------------------------------------------------------------
# Dynamic ControlNet Slot Helpers
# -----------------------------------------------------------------------------

def _controlnet_choices_for_slot(slot_index: int) -> List[str]:
    choices = folder_paths.get_filename_list("controlnet")
    if int(slot_index) == 1:
        return ["DISABLED"] + choices
    return ["DISABLED", "SHARE_PREVIOUS"] + choices


def slot_count_widget(max_slots: int = MAX_CONTROLNET_SLOTS, default: int = 1):
    return (
        "INT",
        {
            "default": int(default),
            "min": 1,
            "max": int(max_slots),
            "step": 1,
            "display": "number",
            "tooltip": (
                "Authoritative active ControlNet slot count. The frontend may "
                "hide rows above this value; the backend ignores them."
            ),
        },
    )


def _float_widget(
    default: float,
    *,
    min_value: float,
    max_value: float,
    step: float,
    tooltip: str,
):
    return (
        "FLOAT",
        {
            "default": float(default),
            "min": float(min_value),
            "max": float(max_value),
            "step": float(step),
            "tooltip": tooltip,
        },
    )


def build_controlnet_setting_slots(max_slots: int = MAX_CONTROLNET_SLOTS) -> Dict[str, Any]:
    """Build all selector/strength/timing/weight widgets for predeclared slots."""

    slots: Dict[str, Any] = {}

    for i in range(1, int(max_slots) + 1):
        idx = f"{i:02d}"

        slots[f"control_net_name_{idx}"] = (
            _controlnet_choices_for_slot(i),
            {
                "tooltip": (
                    "ControlNet selector. DISABLED bypasses this slot. "
                    "SHARE_PREVIOUS reuses the last valid promoted ControlNet."
                )
            },
        )
        slots[f"strength_{idx}"] = _float_widget(
            1.0,
            min_value=0.0,
            max_value=10.0,
            step=0.01,
            tooltip="ControlNet conditioning strength for this slot.",
        )
        slots[f"start_{idx}"] = _float_widget(
            0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            tooltip="Start percent for this ControlNet slot.",
        )
        slots[f"end_{idx}"] = _float_widget(
            1.0,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            tooltip="End percent for this ControlNet slot.",
        )
        slots[f"weight_{idx}"] = _float_widget(
            1.0,
            min_value=-10.0,
            max_value=10.0,
            step=0.01,
            tooltip="Fusion weight for this ControlNet slot.",
        )

    return slots


def build_controlnet_image_inputs(max_slots: int = MAX_CONTROLNET_SLOTS) -> Dict[str, Any]:
    """Build optional lazy IMAGE inputs for all predeclared slots."""

    return {
        f"image_{i:02d}": (
            "IMAGE",
            {
                "lazy": True,
                "tooltip": (
                    "Control image for this slot. This input is lazy so "
                    "inactive or hidden slots do not need to evaluate upstream "
                    "preprocessors."
                ),
            },
        )
        for i in range(1, int(max_slots) + 1)
    }


def normalize_slot_count(slot_count: Any, max_slots: int = MAX_CONTROLNET_SLOTS) -> int:
    try:
        count = int(slot_count)
    except (TypeError, ValueError):
        count = 1
    return max(1, min(int(max_slots), count))


def _float_from_kwargs(kwargs: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(kwargs.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _collect_slots(kwargs: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []

    for i in range(1, int(count) + 1):
        idx = f"{i:02d}"
        slots.append(
            {
                "i": i,
                "image": kwargs.get(f"image_{idx}"),
                "strength": _float_from_kwargs(kwargs, f"strength_{idx}", 1.0),
                "start": _float_from_kwargs(kwargs, f"start_{idx}", 0.0),
                "end": _float_from_kwargs(kwargs, f"end_{idx}", 1.0),
                "weight": _float_from_kwargs(kwargs, f"weight_{idx}", 1.0),
                "name": kwargs.get(f"control_net_name_{idx}", "DISABLED"),
            }
        )

    return slots


def _slot_early_bypassed(slot: Dict[str, Any]) -> bool:
    return (
        slot.get("image") is None
        or float(slot.get("strength", 1.0)) == 0.0
        or float(slot.get("weight", 1.0)) == 0.0
        or (float(slot.get("end", 1.0)) - float(slot.get("start", 0.0))) <= 0.0
    )


def _slot_numeric_bypassed(slot: Dict[str, Any]) -> bool:
    """Early-bypass checks that can be evaluated before lazy images exist."""

    return (
        float(slot.get("strength", 1.0)) == 0.0
        or float(slot.get("weight", 1.0)) == 0.0
        or (float(slot.get("end", 1.0)) - float(slot.get("start", 0.0))) <= 0.0
    )


def _hidden_slot_numbers(count: int, max_slots: int = MAX_CONTROLNET_SLOTS) -> List[str]:
    return [str(i) for i in range(int(count) + 1, int(max_slots) + 1)]


def _print_slot_summary(
    *,
    count: int,
    active: Sequence[Any],
    inactive_visible: Sequence[Any],
    max_slots: int = MAX_CONTROLNET_SLOTS,
) -> None:
    ignored = _hidden_slot_numbers(count, max_slots)
    print(
        f"[JLC-Orchestrator-Adv-Dynamic] slot_count={count} | "
        f"Active: {', '.join(map(str, active)) or 'none'} | "
        f"Inactive visible: {', '.join(map(str, inactive_visible)) or 'none'} | "
        f"Ignored hidden: {', '.join(ignored) or 'none'}"
    )


# -----------------------------------------------------------------------------
# Dynamic Advanced Orchestrator Node
# -----------------------------------------------------------------------------
class JLC_DynamicControlNetOrchestratorAdvanced:
    """
    Single dynamic Advanced ControlNet Orchestrator.

    Backend contract:
        - Only slots 1..slot_count are inspected.
        - Slots above slot_count are ignored even if serialized values exist.
        - DISABLED is a hard bypass.
        - SHARE_PREVIOUS only works after a previous visible slot successfully
          passed early-bypass validation and promoted a base ControlNet.
        - Inactive slots do not promote.
        - Single active ControlNet uses the prior native ApplyAdvanced-style
          fallback path.
        - Multi-ControlNet execution uses JLC_ComposedControlNet unchanged.
    """

    FUNCTION = "orchestrate"
    CATEGORY = "conditioning/controlnet"

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")

    MAX_CONTROLNETS = MAX_CONTROLNET_SLOTS
    NUMBER_OF_MODELS = MAX_CONTROLNET_SLOTS

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        required = {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "slot_count": slot_count_widget(cls.MAX_CONTROLNETS, default=1),
            **build_controlnet_setting_slots(cls.MAX_CONTROLNETS),
        }

        optional = {
            "alpha": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": (
                        "Per-slot order falloff/exponent. Preserved from the "
                        "existing Advanced Orchestrator. alpha=1 keeps raw "
                        "slot weights."
                    ),
                },
            ),
            **build_controlnet_image_inputs(cls.MAX_CONTROLNETS),
        }

        return {"required": required, "optional": optional}

    def check_lazy_status(self, positive, negative, vae, slot_count=1, alpha=1.0, **kwargs):
        """
        Request only the lazy image inputs needed by visible, potentially active slots.

        This mirrors the promotion rules without loading ControlNet models:
        - DISABLED slots do not need images.
        - Numeric-bypassed slots do not need images and do not promote.
        - SHARE_PREVIOUS slots need images only after a previous visible slot has
          already promoted successfully.
        - Because lazy status may be called repeatedly, a later SHARE_PREVIOUS
          slot can become needed after an earlier named slot's image is evaluated.
        """

        count = normalize_slot_count(slot_count, self.MAX_CONTROLNETS)
        slots = _collect_slots(kwargs, count)
        needed: List[str] = []
        has_previous_valid_model = False

        for slot in slots:
            idx = f"{slot['i']:02d}"
            image_key = f"image_{idx}"
            name = slot["name"]

            if not name or name == "DISABLED":
                continue

            if _slot_numeric_bypassed(slot):
                continue

            if name == "SHARE_PREVIOUS" and not has_previous_valid_model:
                continue

            if slot["image"] is None:
                needed.append(image_key)
                continue

            # The slot now has a usable image and passed numeric checks, so it
            # would pass early bypass and promote. Named slots promote their own
            # model; SHARE_PREVIOUS re-promotes the inherited current_base.
            has_previous_valid_model = True

        return needed

    # ------------------------------------------------------------
    # 🧠 MAIN LOGIC
    # ------------------------------------------------------------
    def orchestrate(self, positive, negative, vae, slot_count=1, alpha=1.0, **kwargs):
        count = normalize_slot_count(slot_count, self.MAX_CONTROLNETS)

        # ------------------------------------------------------------
        # Phase 1 — Resolve slot pairing
        # ------------------------------------------------------------
        slots = _collect_slots(kwargs, count)
        resolved = []
        current_base = None

        # ----------------------------------------------
        # Model Resolution Loop
        # ----------------------------------------------
        for slot in slots:

            name = slot["name"]

            # ----------------------------------------
            # 🚫 HARD BYPASS
            # ----------------------------------------
            if not name or name == "DISABLED":
                continue

            # ----------------------------------------
            # 🔄 SHARE PREVIOUS
            # ----------------------------------------
            if name == "SHARE_PREVIOUS":
                if current_base is None:
                    continue
                candidate_base = current_base

            # ----------------------------------------
            # 🔄 LOAD / CACHE
            # ----------------------------------------
            else:
                path = folder_paths.get_full_path_or_raise("controlnet", name)
                candidate_base = get_controlnet(path)

            # ------------------------------------------------------------
            # 🚫 EARLY BYPASS (CRITICAL)
            # ------------------------------------------------------------
            if _slot_early_bypassed(slot):
                continue

            current_base = candidate_base

            resolved.append(
                {
                    **slot,
                    "base": candidate_base,
                }
            )

        # --------------------------------------------
        # Active vs. Inactive Slot Classification
        # --------------------------------------------
        active = [str(item["i"]) for item in resolved]
        inactive_visible = [str(s["i"]) for s in slots if str(s["i"]) not in active]

        _print_slot_summary(
            count=count,
            active=active,
            inactive_visible=inactive_visible,
            max_slots=self.MAX_CONTROLNETS,
        )

        if DEBUG:
            info = cache_info()
            controlnet_entries = [
                entry for entry in info.get("entries", []) if entry.get("family") == "controlnet"
            ]
            print(
                f"[JLC-Orchestrator-Adv-Dynamic] ControlNet cache: "
                f"{len(controlnet_entries)}/{info.get('family_capacity', {}).get('controlnet', '?')} resident"
            )

        if not resolved:
            return (positive, negative)

        # ------------------------------------------------------------
        # Phase 2 — 🧱 Build independent ControlNets
        # ------------------------------------------------------------
        prepared_cnets = []
        weights = []

        for item in resolved:
            cnet_base = item["base"]
            image = item["image"]
            strength = item["strength"]
            weight = item["weight"]

            control_hint = image.movedim(-1, 1)

            cnet = (
                cnet_base
                .copy()
                .set_cond_hint(
                    control_hint,
                    strength,
                    (item["start"], item["end"]),
                    vae=vae,
                )
            )

            prepared_cnets.append(cnet)
            weights.append(weight)

        # ------------------------------------------------------------
        # 🟡 Special Case — Single ControlNet (bypass composition)
        # ------------------------------------------------------------
        if len(prepared_cnets) == 1:
            single_cnet = prepared_cnets[0]

            def inject_single(conditioning):
                out = []
                for t in conditioning:
                    d = copy.deepcopy(t[1])

                    prev_cnet = d.get('control', None)

                    # replicate Comfy's stock ApplyAdvanced behavior
                    c_net = single_cnet.copy()
                    c_net.set_previous_controlnet(prev_cnet)

                    d['control'] = c_net
                    d['control_apply_to_uncond'] = False

                    out.append([t[0], d])
                return out

            return (inject_single(positive), inject_single(negative))

        # ------------------------------------------------------------
        # Phase 3 — 🎶 Compose
        # ------------------------------------------------------------
        final_weights = [
            w * (alpha ** i)
            for i, w in enumerate(weights)
        ]

        composed = JLC_ComposedControlNet(prepared_cnets, final_weights)

        # ------------------------------------------------------------
        #  Phase 4 — 💉 Inject into conditioning
        # ------------------------------------------------------------
        def inject(conditioning):
             out = []
             for t in conditioning:
                 d = copy.deepcopy(t[1])
                 d["control"] = composed
                 d["control_apply_to_uncond"] = False
                 out.append([t[0], d])
             return out

        return (inject(positive), inject(negative))


# -----------------------------------------------------------------------------
# Node Registration
# -----------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "JLC_DynamicControlNetOrchestratorAdvanced": JLC_DynamicControlNetOrchestratorAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_DynamicControlNetOrchestratorAdvanced": "\u2003JLC ControlNet Orchestrator - Advanced Dynamic",
}
