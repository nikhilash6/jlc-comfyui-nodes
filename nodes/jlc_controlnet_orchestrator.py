"""
JLC ControlNet Orchestrator

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
    - The **JLC ControlNet Orchestrator** introduces an experimental,
      fully non-recursive ControlNet execution model that departs from
      ComfyUI’s native chained (`previous_controlnet`) architecture.

    - Instead of sequential dependency and inherited state, this node:
            • Accepts multiple ControlNet inputs (slot-based)
            • Creates isolated instances via `.copy()` (no mutation)
            • Applies independent conditioning per slot
            • Executes each ControlNet against the same latent input
            • Combines outputs via weighted additive fusion

    - Composition is defined as:
            combined = Σ (w_i · C_i(x))

      where:
            • C_i(x) = independent ControlNet outputs
            • w_i = user-defined weights (supports negative values)

    - This approach:
            • Eliminates recursive chaining and state inheritance
            • Guarantees zero cross-contamination between ControlNets
            • Provides deterministic, interpretable multi-CNet behavior
            • Preserves compatibility with ComfyUI conditioning pipelines

    - Philosophical deviation:
        - ControlNets are treated as **independent operators**, not
          linked transformations. Interaction occurs only at the level
          of output aggregation, not during execution.

    - ⚠️ Experimental Code:
        - This node represents an ongoing exploration of alternative
          ControlNet execution strategies.
        - Behavior, performance characteristics, and API may evolve.
        - Intended for advanced users and controlled testing scenarios.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

MANIFEST = {
    "name": "JLC ControlNet Orchestrator",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": (
        "Node that implements a novel non-recursive, orchestrated ControlNet composition approach, "
        "closely approximating native ControlNet interaction dynamics without recursive chaining. "
        "Avoids explicit chain construction while preserving interaction behavior, enabling more "
        "predictable performance, reduced peak memory pressure, and improved stability across "
        "multi-ControlNet workflows."
    ),
}


import torch

DEBUG = True

# ------------------------------------------------------------
# 🧠 Core Fusion Wrapper (reused from your Composition node)
# ------------------------------------------------------------
class JLC_ComposedControlNet:
    def __init__(self, controlnets, weights):
        self.controlnets = controlnets
        self.weights = weights
        self.previous_controlnet = None
        self.extra_hooks = None

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):

        combined = None

        for idx, (cnet, w) in enumerate(zip(self.controlnets, self.weights)):

            if cnet is None or w == 0:
                continue

            # ------------------------------------------------------------
            # 🔵 Phase 1 — Execute ControlNet
            # ------------------------------------------------------------
            out = cnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

            if out is None:
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # ------------------------------------------------------------
            # 🟢 Phase 2 — First ownership (clone)
            # ------------------------------------------------------------
            if combined is None:
                combined = {}

                for key, out_list in out.items():
                    new_list = [None] * len(out_list)

                    for i, v in enumerate(out_list):
                        if v is None:
                            continue

                        owned = v.clone()

                        if w != 1.0:
                            owned.mul_(w)

                        new_list[i] = owned

                    combined[key] = new_list

            # ------------------------------------------------------------
            # 🟣 Phase 3 — Accumulation
            # ------------------------------------------------------------
            else:

                for key, out_list in out.items():

                    if key not in combined:
                        combined[key] = [None] * len(out_list)

                    combined_list = combined[key]

                    if len(combined_list) < len(out_list):
                        combined_list.extend([None] * (len(out_list) - len(combined_list)))

                    for i, v in enumerate(out_list):
                        if v is None:
                            continue

                        dst = combined_list[i]

                        if dst is None:
                            owned = v.clone()

                            if w != 1.0:
                                owned.mul_(w)

                            combined_list[i] = owned
                        else:
                            dst.add_(v, alpha=w)

                    del out_list

            # ------------------------------------------------------------
            # 🔴 Phase 4 — Release
            # ------------------------------------------------------------
            del out

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return combined
    

# ------------------------------------------------------------
# 🧠 KSampler Compatibility — ControlNet Interface Passthrough
# ------------------------------------------------------------
# These methods forward required ControlNet interface calls to all
# underlying ControlNet instances, preserving full compatibility
# with ComfyUI's execution pipeline (hooks, model loading,
# memory estimation, and lifecycle management).

    def get_extra_hooks(self):
        hooks = []
        for cnet in self.controlnets:
            if hasattr(cnet, "get_extra_hooks"):
                hooks += cnet.get_extra_hooks()
        return hooks


    def get_models(self):
        models = []
        for cnet in self.controlnets:
            if hasattr(cnet, "get_models"):
                models += cnet.get_models()
        return models


    def inference_memory_requirements(self, dtype):
        max_req = 0
        for cnet in self.controlnets:
            if cnet is None:
                continue
            if hasattr(cnet, "inference_memory_requirements"):
                req = cnet.inference_memory_requirements(dtype)
                if req is not None:
                    max_req = max(max_req, req)
        return max_req


    def pre_run(self, model, percent_to_timestep_function):
        for cnet in self.controlnets:
            if hasattr(cnet, "pre_run"):
                cnet.pre_run(model, percent_to_timestep_function)


    def cleanup(self):
        for cnet in self.controlnets:
            if hasattr(cnet, "cleanup"):
                cnet.cleanup()


# ------------------------------------------------------------
# 🎯 Main Node
# ------------------------------------------------------------
class JLC_ControlNetOrchestrator:

    FUNCTION = "orchestrate"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
            "optional": {
                # --- SLOT 1 ---
                "image_01": ("IMAGE",),
                "control_net_01": ("CONTROL_NET", {
                    "tooltip": "Primary ControlNet for Slot 1"
                }),
                "enabled_01": ("BOOLEAN", {"default": True}),
                "strength_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_01": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # --- SLOT 2 ---
                "image_02": ("IMAGE",),
                "control_net_02": ("CONTROL_NET", {
                    "tooltip": "If empty, reuses previous ControlNet (implicit SHARE)"
                }),
                "enabled_02": ("BOOLEAN", {"default": True}),
                "strength_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_02": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # --- SLOT 3 ---
                "image_03": ("IMAGE",),
                "control_net_03": ("CONTROL_NET", {
                    "tooltip": "If empty, reuses previous ControlNet (implicit SHARE)"
                }),
                "enabled_03": ("BOOLEAN", {"default": True}),
                "strength_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_03": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # --- Order-dependent parameter. Breaks slot-invariance ---
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Order bias. 1.0 = neutral. <1 favors earlier slots. >1 favors later slots. Negative values invert influence."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")

    # ------------------------------------------------------------
    # 🧠 MAIN LOGIC
    # ------------------------------------------------------------
    def orchestrate(self, positive, negative, vae, **kwargs):

        # ------------------------------------------------------------
        # Phase 1 — Resolve slot pairing
        # ------------------------------------------------------------
        resolved = []

        current_cnet = None

        for i in range(1, 4):
            idx = f"{i:02d}"
            image = kwargs.get(f"image_{idx}")
            enabled = kwargs.get(f"enabled_{idx}", True)
            cnet = kwargs.get(f"control_net_{idx}")
            strength = kwargs.get(f"strength_{idx}", 1.0)
            start = kwargs.get(f"start_{idx}", 0.0)
            end = kwargs.get(f"end_{idx}", 1.0)
            weight = kwargs.get(f"weight_{idx}", 1.0)
            alpha = kwargs.get("alpha", 1.0)

            # Checking only the incoming cnet
            incoming_cnet = cnet if cnet else current_cnet


            # ------------------------------------------------------------
            # 🚫 EARLY BYPASS (CRITICAL)
            # ------------------------------------------------------------
            if (
                not enabled
                or incoming_cnet is None
                or image is None
                or strength == 0
                or weight == 0
                or (end - start) <= 0
            ):
                continue

            current_cnet = incoming_cnet

            resolved.append({
                "cnet": current_cnet,
                "image": image,
                "strength": strength,
                "start": start,
                "end": end,
                "weight": weight,
            })

        debug = DEBUG

        if debug:
            active = []
            inactive = []

            debug_current_cnet = None

            for i in range(1, 4):
                idx = f"{i:02d}"

                enabled = kwargs.get(f"enabled_{idx}", True)
                cnet = kwargs.get(f"control_net_{idx}")
                image = kwargs.get(f"image_{idx}")
                strength = kwargs.get(f"strength_{idx}", 1.0)
                start = kwargs.get(f"start_{idx}", 0.0)
                end = kwargs.get(f"end_{idx}", 1.0)
                weight = kwargs.get(f"weight_{idx}", 1.0)
                
                incoming_cnet = cnet if cnet else debug_current_cnet

                if (
                    enabled
                    and (incoming_cnet is not None)
                    and image is not None
                    and strength != 0
                    and weight != 0
                    and (end - start) > 0
                ):
                    active.append(str(i))
                    debug_current_cnet = incoming_cnet  # mirror promotion
                else:
                    inactive.append(str(i))

            print(f"[JLC-Orchestrator] Active: {', '.join(active) or 'none'} | Inactive: {', '.join(inactive)}")         

        if not resolved:
            return (positive, negative)

        # ------------------------------------------------------------
        # Phase 2 — Build independent ControlNets
        # ------------------------------------------------------------
        prepared_cnets = []
        weights = []

        for item in resolved:
            cnet_base = item["cnet"]
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
                    d = t[1].copy()

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
        # Phase 3 — Compose
        # ------------------------------------------------------------
        final_weights = [
            w * (alpha ** i)
            for i, w in enumerate(weights)
        ]
        
        composed = JLC_ComposedControlNet(prepared_cnets, final_weights)

        # ------------------------------------------------------------
        # Phase 4 — Inject into conditioning
        # ------------------------------------------------------------
        def inject(conditioning):
            out = []
            for t in conditioning:
                d = t[1].copy()
                d["control"] = composed
                d["control_apply_to_uncond"] = False
                out.append([t[0], d])
            return out

        return (inject(positive), inject(negative))