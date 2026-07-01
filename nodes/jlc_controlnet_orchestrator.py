"""
JLC ControlNet Orchestrator
---------------------------

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
        • multi-ControlNet composition and orchestration

- Node Purpose
  - The **JLC ControlNet Orchestrator** builds a non-recursive multi-ControlNet
    composition directly from wired ControlNet and hint-image slots.

  - Instead of asking the user to construct a native `previous_controlnet`
    chain first, the node:
        • accepts multiple wired ControlNet inputs;
        • supports dynamic visible slot count;
        • optionally reuses the previous wired ControlNet when a later slot
          has no new ControlNet connected;
        • applies independent hint images, strengths, and timestep ranges per
          slot;
        • creates isolated per-slot ControlNet copies;
        • routes one active ControlNet through native ComfyUI chaining;
        • routes two or more active ControlNets through the shared JLC
          non-recursive composed wrapper.

  - Multi-ControlNet composition is defined as:
        combined = Σ (w_i · α^i) · C_i(x)

    where:
        • C_i(x) is the independent output of active slot i;
        • w_i is the user-defined slot weight;
        • α is an order-bias term applied across active slots.

  - Slot validation is intentionally stricter than optional ComfyUI socket
    validation.  A slot that is effectively active must have a hint image:
        • enabled;
        • ControlNet available directly or by previous-slot reuse;
        • non-zero strength;
        • non-zero weight;
        • valid start/end range.

    If such a slot has no image connected, the node raises a runtime error so
    ComfyUI highlights the miswired Orchestrator instead of silently skipping
    the slot.

  - This node performs no internal model loading or cache management.  It is
    designed for explicit wired ControlNet workflows where model residency is
    controlled elsewhere in the graph.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in the core **ComfyUI**
    project:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

from .jlc_controlnet_nonrecursive_core import (
        clear_multigpu_clone_state,
        compose_or_native_fallback,
        safe_cnet_name,
)


MANIFEST = {
    "name": "JLC ControlNet Orchestrator",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Wired multi-ControlNet orchestrator using JLC non-recursive weighted "
        "fusion for two or more active slots and native ControlNet chaining for "
        "a single active slot. Supports dynamic visible slots, previous-ControlNet "
        "reuse, strict missing-hint validation, and no internal model loading."
    ),
}

MAX_SLOTS = 10
DEBUG = True


class JLC_ControlNetOrchestrator:
    FUNCTION = "orchestrate"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "slot_count": ("INT", {
                "default": 3,
                "min": 1,
                "max": MAX_SLOTS,
                "step": 1,
                "tooltip": "Number of visible/active wired ControlNet slots. Backend ignores slots above this count.",
            }),
        }

        for i in range(1, MAX_SLOTS + 1):
            idx = f"{i:02d}"
            optional[f"control_net_{idx}"] = ("CONTROL_NET", {
                "tooltip": "ControlNet for this slot. Empty slots after Slot 1 reuse the previous active ControlNet if available."
            })
            optional[f"image_{idx}"] = ("IMAGE", {
                "tooltip": "Control image for this slot. Required when this slot is enabled and otherwise active."
            })
            optional[f"enabled_{idx}"] = ("BOOLEAN", {"default": True})
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

    def orchestrate(self, positive, negative, vae, slot_count=3, alpha=1.0, **kwargs):
        slot_count = max(1, min(MAX_SLOTS, int(slot_count)))

        resolved = []
        current_cnet = None
        inactive_reasons = {}

        for i in range(1, slot_count + 1):
            idx = f"{i:02d}"
            enabled = kwargs.get(f"enabled_{idx}", True)
            cnet = kwargs.get(f"control_net_{idx}")
            image = kwargs.get(f"image_{idx}")
            strength = kwargs.get(f"strength_{idx}", 1.0)
            start = kwargs.get(f"start_{idx}", 0.0)
            end = kwargs.get(f"end_{idx}", 1.0)
            weight = kwargs.get(f"weight_{idx}", 1.0)

            incoming_cnet = cnet if cnet is not None else current_cnet
            has_valid_range = (end - start) > 0
            has_meaningful_control = (
                enabled
                and incoming_cnet is not None
                and strength != 0
                and weight != 0
                and has_valid_range
            )

            if has_meaningful_control and image is None:
                source = "direct ControlNet input" if cnet is not None else "reused previous ControlNet"
                raise RuntimeError(
                    f"JLC ControlNet Orchestrator: slot {i} is active ({source}, "
                    f"strength={strength}, weight={weight}, range=({start}, {end})) "
                    f"but image_{idx} is not connected. Connect a hint image, disable the slot, "
                    "set strength/weight to 0, or set an empty range."
                )

            if not enabled:
                inactive_reasons[i] = "disabled"
                continue
            if incoming_cnet is None:
                inactive_reasons[i] = "no_controlnet"
                continue
            if strength == 0:
                inactive_reasons[i] = "strength_zero"
                continue
            if weight == 0:
                inactive_reasons[i] = "weight_zero"
                continue
            if not has_valid_range:
                inactive_reasons[i] = "empty_range"
                continue
            if image is None:
                # Defensive fallback; normally unreachable because the active
                # missing-image case is raised above.
                inactive_reasons[i] = "missing_image"
                continue

            current_cnet = incoming_cnet
            resolved.append({
                "slot": i,
                "base": current_cnet,
                "image": image,
                "strength": strength,
                "start": start,
                "end": end,
                "weight": weight,
            })

        if DEBUG:
            active = [str(item["slot"]) for item in resolved]
            inactive = [str(i) for i in range(1, slot_count + 1) if str(i) not in active]
            reason_text = ", ".join(
                f"{slot}:{inactive_reasons.get(slot, 'inactive')}" for slot in range(1, slot_count + 1)
                if str(slot) not in active
            )
            print(
                f"[JLC-Orchestrator] slot_count={slot_count} "
                f"active={', '.join(active) or 'none'} inactive={', '.join(inactive) or 'none'} "
                f"inactive_reasons={reason_text or 'none'} alpha={alpha}"
            )

        if not resolved:
            return (positive, negative)

        prepared_cnets = []
        weights = []
        debug_names = []

        for item in resolved:
            control_hint = item["image"].movedim(-1, 1)
            cnet = (
                item["base"]
                .copy()
                .set_cond_hint(
                    control_hint,
                    item["strength"],
                    (item["start"], item["end"]),
                    vae=vae,
                )
            )
            clear_multigpu_clone_state(cnet)
            prepared_cnets.append(cnet)
            weights.append(item["weight"])

            name = f"slot_{item['slot']:02d}:{safe_cnet_name(cnet)}"
            debug_names.append(name)

            if DEBUG:
                print(
                    f"[JLC-Orchestrator] prepared slot={item['slot']} "
                    f"name={name} strength={item['strength']} "
                    f"range=({item['start']}, {item['end']}) weight={item['weight']}"
                )

        return compose_or_native_fallback(
            positive,
            negative,
            prepared_cnets,
            weights,
            alpha=alpha,
            debug_label="JLC-Orchestrator",
            debug_names=debug_names,
        )
