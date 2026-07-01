"""
JLC ControlNet Orchestrator
---------------------------

External-CONTROL_NET-slot Orchestrator for JLC non-recursive ControlNet
composition.

This pass removes the older inline wrapper duplication and delegates all
non-recursive composition behavior to jlc_controlnet_nonrecursive_core. The core
fusion math remains the April linearized non-recursive algorithm.
"""

from __future__ import annotations

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

from .jlc_controlnet_nonrecursive_core import (
    DEBUG,
    compose_or_native_fallback,
    prepare_controlnet_copy,
    safe_cnet_name,
)

MANIFEST = {
    "name": "JLC ControlNet Orchestrator",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "External-slot multi-ControlNet orchestrator using the shared JLC "
        "non-recursive weighted fusion core. Accepts ControlNet objects from "
        "upstream loaders, prepares independent per-slot copies, preserves native "
        "single-ControlNet routing when mathematically equivalent, and composes "
        "multi-ControlNet workflows through one sampler-facing wrapper."
    ),
}

MAX_SLOTS = 3


class JLC_ControlNetOrchestrator:
    FUNCTION = "orchestrate"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}

        for i in range(1, MAX_SLOTS + 1):
            idx = f"{i:02d}"
            optional[f"image_{idx}"] = ("IMAGE", {
                "tooltip": "Control image for this slot. Slot is skipped if this is not connected."
            })
            optional[f"control_net_{idx}"] = ("CONTROL_NET", {
                "tooltip": "ControlNet for this slot. Empty slots reuse the previous active ControlNet where possible."
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

    def orchestrate(self, positive, negative, vae, alpha=1.0, **kwargs):
        resolved = []
        current_cnet = None
        inactive_reasons = {}

        for i in range(1, MAX_SLOTS + 1):
            idx = f"{i:02d}"
            enabled = kwargs.get(f"enabled_{idx}", True)
            image = kwargs.get(f"image_{idx}")
            incoming = kwargs.get(f"control_net_{idx}")
            strength = kwargs.get(f"strength_{idx}", 1.0)
            start = kwargs.get(f"start_{idx}", 0.0)
            end = kwargs.get(f"end_{idx}", 1.0)
            weight = kwargs.get(f"weight_{idx}", 1.0)

            # Empty ControlNet slot means implicit SHARE_PREVIOUS, matching the
            # original simple Orchestrator behavior.
            cnet_base = incoming if incoming is not None else current_cnet

            if not enabled:
                inactive_reasons[i] = "disabled"
                continue
            if cnet_base is None:
                inactive_reasons[i] = "no_controlnet"
                continue
            if image is None:
                inactive_reasons[i] = "no_image"
                continue
            if strength == 0:
                inactive_reasons[i] = "strength_zero"
                continue
            if weight == 0:
                inactive_reasons[i] = "weight_zero"
                continue
            if (end - start) <= 0:
                inactive_reasons[i] = "empty_range"
                continue

            current_cnet = cnet_base
            resolved.append({
                "slot": i,
                "base": cnet_base,
                "image": image,
                "strength": strength,
                "start": start,
                "end": end,
                "weight": weight,
            })

        if DEBUG:
            active = [str(item["slot"]) for item in resolved]
            inactive = [str(i) for i in range(1, MAX_SLOTS + 1) if str(i) not in active]
            reason_text = ", ".join(
                f"{slot}:{inactive_reasons.get(slot, 'inactive')}"
                for slot in range(1, MAX_SLOTS + 1)
                if str(slot) not in active
            )
            print(
                f"[JLC-Orchestrator] active={', '.join(active) or 'none'} "
                f"inactive={', '.join(inactive) or 'none'} "
                f"inactive_reasons={reason_text or 'none'} alpha={alpha}"
            )

        if not resolved:
            return (positive, negative)

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
            debug_names.append(f"slot_{item['slot']:02d}:{safe_cnet_name(cnet)}")

            if DEBUG:
                print(
                    f"[JLC-Orchestrator] prepared slot={item['slot']} "
                    f"type={safe_cnet_name(cnet)} strength={item['strength']} "
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


NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetOrchestrator": JLC_ControlNetOrchestrator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetOrchestrator": "JLC ControlNet Orchestrator",
}
