"""
JLC ControlNet Orchestrator (Advanced)
--------------------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for
    image-generation pipelines, particularly:
        • Flux-based workflows
        • LoRA experimentation
        • advanced inpainting / outpainting pipelines
        • multi-ControlNet composition and orchestration

- Node Purpose
  - The **JLC ControlNet Orchestrator (Advanced)** is the integrated,
    internal-loader interface for building weighted multi-ControlNet
    workflows with the JLC non-recursive composition architecture.

  - The node provides:
        • up to eight dynamically exposed ControlNet slots
        • internal model selection through the shared JLC model cache
        • `SHARE_PREVIOUS` reuse of the most recently selected model
        • independent hint, strength, start, end, and weight controls
          for each slot
        • strict slot validation before any ControlNet model is loaded
        • optional order bias through the global `alpha` parameter

  - Each active slot:
        • resolves or reuses a raw cached ControlNet base model
        • creates an isolated per-run ControlNet copy
        • applies its own hint image, strength, activation range, and VAE
        • remains independent of the other prepared ControlNet instances

  - Multi-ControlNet execution uses the shared JLC linearized,
    non-recursive fusion core:

        combined = Σ [(w_i · alpha^i) · C_i(x)]

    where:
        • C_i(x) is the output of ControlNet slot `i`, evaluated
          independently against the same sampler state
        • w_i is the user-defined slot weight
        • alpha^i applies the optional order bias
        • negative weights and negative alpha values are supported

  - The fusion core:
        • does not build a recursive `previous_controlnet` chain
          between active slots
        • does not mutate cached base ControlNet objects
        • does not use `deepcopy`
        • evaluates prepared ControlNets independently
        • clones tensors only when taking ownership of output data
        • accumulates subsequent outputs in-place
        • presents one ControlNet-compatible wrapper to the ComfyUI sampler

  - When only one effective ControlNet remains at a final weight of 1.0,
    the node uses the mathematically equivalent native single-ControlNet
    path instead of creating a composed wrapper.

  - Runtime Integration
    - The node exposes child models, hooks, lifecycle methods, and inference
      memory requirements through the interfaces expected by current ComfyUI
      sampler and model-management paths.

    - It is designed to cooperate with normal ComfyUI model management and
      DynamicVRAM behavior. It does not replace ComfyUI loading, offloading,
      patching, or device-residency policy.

    - The shared JLC cache stores raw ControlNet base objects only.
      Per-run conditioning state is applied exclusively to isolated copies.

  - Relationship to JLC ControlNet Composition
    - **Orchestrator Advanced** is the compact, integrated workflow.

    - **Apply Advanced → Composition** is the corresponding modular workflow.

    - Both use the same validated non-recursive weighted-fusion algorithm;
      neither path is mathematically subordinate to the other.

  - MultiGPU Scope
    - The wrapper includes compatibility shunts required by current ComfyUI
      interfaces but does not implement real MultiGPU ControlNet cloning.

    - Use this node as a single-device ControlNet orchestration path unless
      explicit MultiGPU support is added in a future implementation.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by and interoperable with the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

import folder_paths
import comfy.controlnet

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

from .jlc_controlnet_nonrecursive_core import (
    DEBUG,
    clear_multigpu_clone_state,
    compose_or_native_fallback,
    prepare_controlnet_copy,
    safe_cnet_name,
)

from .engines.jlc_model_cache_core import (
    get_controlnet_cache_capacity,
    get_or_load_model,
    make_controlnet_cache_key,
)

MANIFEST = {
    "name": "JLC ControlNet Orchestrator (Advanced)",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Integrated internal-loader multi-ControlNet orchestrator using the "
        "shared JLC model cache and validated non-recursive weighted-fusion "
        "core. Supports dynamic slots, SHARE_PREVIOUS model reuse, strict "
        "preflight before model loading, independent per-slot conditioning, "
        "order-biased weights, native single-ControlNet routing when "
        "mathematically equivalent, and current ComfyUI sampler and "
        "model-management compatibility."
    ),
}

MAX_SLOTS = 8
DISABLED = "DISABLED"
SHARE_PREVIOUS = "SHARE_PREVIOUS"


def _default_cache_size():
    try:
        return max(0, int(get_controlnet_cache_capacity()))
    except Exception:
        return 2


def _load_controlnet_with_shared_cache(control_net_name, cache_size=None):
    controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
    key = make_controlnet_cache_key(controlnet_path)

    def loader():
        if DEBUG:
            print(f"[JLC-ControlNet Cache] loading ControlNet: {control_net_name}")
        cnet = comfy.controlnet.load_controlnet(controlnet_path)
        if cnet is None:
            raise RuntimeError(f"Invalid ControlNet model file: {control_net_name}")
        clear_multigpu_clone_state(cnet)
        return cnet

    max_loaded_for_family = None
    if cache_size is not None:
        max_loaded_for_family = max(0, int(cache_size))

    return get_or_load_model(
        key,
        loader,
        family="controlnet",
        model_path=controlnet_path,
        role="controlnet",
        policy="lru_family_capacity",
        max_loaded_for_family=max_loaded_for_family,
        metadata={"control_net_name": control_net_name},
    )


class JLC_ControlNetOrchestratorAdvanced:
    FUNCTION = "orchestrate"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        controlnet_names = folder_paths.get_filename_list("controlnet")

        optional = {
            "slot_count": ("INT", {
                "default": 3,
                "min": 1,
                "max": MAX_SLOTS,
                "step": 1,
                "tooltip": "Number of visible/active internal ControlNet slots. Backend ignores slots above this count.",
            }),
            "controlnet_cache_size": ("INT", {
                "default": _default_cache_size(),
                "min": 0,
                "max": 10,
                "step": 1,
                "advanced": True,
                "tooltip": "Shared JLC ControlNet cache capacity. 0 means evict/prevent resident cached ControlNets.",
            }),
        }

        for i in range(1, MAX_SLOTS + 1):
            idx = f"{i:02d}"
            choices = [DISABLED] + ([] if i == 1 else [SHARE_PREVIOUS]) + controlnet_names
            optional[f"control_net_name_{idx}"] = (choices, {
                "tooltip": "ControlNet model for this slot. SHARE_PREVIOUS reuses the last selected model."
            })
            optional[f"image_{idx}"] = ("IMAGE", {
                "tooltip": "Control image for this slot. Required when this slot is selected/reused and otherwise active."
            })
            optional[f"strength_{idx}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            optional[f"start_{idx}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001})
            optional[f"end_{idx}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            optional[f"weight_{idx}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        optional["alpha"] = ("FLOAT", {
            "default": 1.0,
            "min": -2.0,
            "max": 2.0,
            "step": 0.01,
            "tooltip": "Order bias. 1.0 = neutral. <1 favors earlier slots. >1 favors later slots. Negative values invert influence.",
        })

        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")

    def orchestrate(self, positive, negative, vae, slot_count=3, controlnet_cache_size=None, alpha=1.0, **kwargs):
        slot_count = max(1, min(MAX_SLOTS, int(slot_count)))
        cache_size = _default_cache_size() if controlnet_cache_size is None else int(controlnet_cache_size)

        # Pass 1: pure slot preflight. Do not touch cache or load model files
        # until all active slots have valid wiring.
        resolved_specs = []
        current_name = None
        inactive_reasons = {}

        for i in range(1, slot_count + 1):
            idx = f"{i:02d}"
            name = kwargs.get(f"control_net_name_{idx}", DISABLED)
            image = kwargs.get(f"image_{idx}")
            strength = kwargs.get(f"strength_{idx}", 1.0)
            start = kwargs.get(f"start_{idx}", 0.0)
            end = kwargs.get(f"end_{idx}", 1.0)
            weight = kwargs.get(f"weight_{idx}", 1.0)

            if name in (None, "", DISABLED):
                inactive_reasons[i] = "disabled"
                continue

            if name == SHARE_PREVIOUS:
                if current_name is None:
                    inactive_reasons[i] = "share_previous_without_model"
                    continue
                resolved_name = current_name
                source = SHARE_PREVIOUS
            else:
                resolved_name = name
                source = "selected"
                # Intentional: SHARE_PREVIOUS tracks the last selected model name,
                # even if this selected slot later becomes inactive due to strength,
                # weight, range, or image preflight.
                current_name = resolved_name

            has_valid_range = (end - start) > 0
            has_meaningful_control = strength != 0 and weight != 0 and has_valid_range

            if not has_meaningful_control:
                if strength == 0:
                    inactive_reasons[i] = "strength_zero"
                elif weight == 0:
                    inactive_reasons[i] = "weight_zero"
                else:
                    inactive_reasons[i] = "empty_range"
                continue

            if image is None:
                raise RuntimeError(
                    f"JLC ControlNet Orchestrator Advanced: slot {i} is active "
                    f"({source} ControlNet '{resolved_name}', strength={strength}, "
                    f"weight={weight}, range=({start}, {end})) but image_{idx} is not connected. "
                    "Connect a hint image, set the slot to DISABLED, set strength/weight to 0, "
                    "or set an empty range."
                )

            resolved_specs.append({
                "slot": i,
                "name": resolved_name,
                "source": source,
                "image": image,
                "strength": strength,
                "start": start,
                "end": end,
                "weight": weight,
            })

        if DEBUG:
            active = [str(item["slot"]) for item in resolved_specs]
            inactive = [str(i) for i in range(1, slot_count + 1) if str(i) not in active]
            reason_text = ", ".join(
                f"{slot}:{inactive_reasons.get(slot, 'inactive')}"
                for slot in range(1, slot_count + 1)
                if str(slot) not in active
            )
            print(
                f"[JLC-Orchestrator-Advanced] slot_count={slot_count} cache_size={cache_size} "
                f"active={', '.join(active) or 'none'} inactive={', '.join(inactive) or 'none'} "
                f"inactive_reasons={reason_text or 'none'} alpha={alpha}"
            )

        if not resolved_specs:
            return (positive, negative)

        # Pass 2: model/cache resolution after preflight.
        resolved = []
        current_base = None
        current_base_name = None

        for spec in resolved_specs:
            resolved_name = spec["name"]
            if current_base is None or current_base_name != resolved_name:
                current_base = _load_controlnet_with_shared_cache(resolved_name, cache_size=cache_size)
                current_base_name = resolved_name

            item = dict(spec)
            item["base"] = current_base
            resolved.append(item)

        prepared_cnets = []
        weights = []
        debug_names = []

        for item in resolved:
            control_hint = item["image"].movedim(-1, 1)
            cnet = prepare_controlnet_copy(
                item["base"],
                control_hint,
                item["strength"],
                (item["start"], item["end"]),
                vae=vae,
            )
            prepared_cnets.append(cnet)
            weights.append(item["weight"])

            name = f"slot_{item['slot']:02d}:{item['name']}:{safe_cnet_name(cnet)}"
            debug_names.append(name)

            if DEBUG:
                print(
                    f"[JLC-Orchestrator-Advanced] prepared slot={item['slot']} "
                    f"name={item['name']} type={safe_cnet_name(cnet)} "
                    f"strength={item['strength']} range=({item['start']}, {item['end']}) "
                    f"weight={item['weight']}"
                )

        return compose_or_native_fallback(
            positive,
            negative,
            prepared_cnets,
            weights,
            alpha=alpha,
            debug_label="JLC-Orchestrator-Advanced",
            debug_names=debug_names,
        )


NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetOrchestratorAdvanced": JLC_ControlNetOrchestratorAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetOrchestratorAdvanced": "JLC ControlNet Orchestrator (Advanced)",
}
