"""
JLC ControlNet Orchestrator (Advanced)

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
    - The **JLC ControlNet Orchestrator (Advanced)** extends the base
      Orchestrator with built-in ControlNet loading and caching,
      eliminating the need for external loader nodes.

    - It preserves the same non-recursive execution model:
            • Independent ControlNet instances per slot
            • `.copy()`-based isolation (no shared state)
            • Execution on the same latent input
            • Streaming weighted fusion of outputs

    - Composition is defined as:
            combined = Σ (w_i · C_i(x))

- Built-in Loader System
    - ControlNets are selected via dropdowns and loaded internally
    - A small persistent cache avoids redundant model loads:
            GLOBAL_CONTROLNET_CACHE[path] → ControlNet

    - This cache:
            • Stores base ControlNet models only
            • Does NOT share execution state across slots
            • Preserves isolation via per-slot `.copy()`

- Execution Model
    - Same guarantees as base node:
            • Non-recursive (no `previous_controlnet`)
            • Independent per-slot conditioning
            • Deterministic fusion (order-invariant when α = 1)

    - Slot resolution semantics:
            • "DISABLED" → slot ignored
            • "SHARE_PREVIOUS" → inherits last valid ControlNet
            • Promotion occurs ONLY after early bypass validation

    - Early bypass conditions:
            • Missing image
            • Zero strength
            • Zero weight
            • Invalid (start, end) interval

    → Ensures:
            ✔ No inactive slot can influence downstream execution
            ✔ No order-dependent contamination via slot reuse

- Critical correctness guarantees:
            • Slot-order invariance (for α = 1)
            • Zero cross-contamination via `.copy()`
            • Cache-safe execution (no shared conditioning state)
            • Single-ControlNet fallback to native Apply semantics

- Philosophical Position
    - Advanced is a strict superset of the base node:
            • Adds ergonomic loader + cache
            • Preserves identical execution semantics

    - ControlNets remain:
            → independent operators
            → combined only at output level

- ⚠️ Experimental Code
    - This node represents a non-canonical formulation of ControlNet
      interaction that diverges from ComfyUI’s native chained execution model.
    - Behavior is stable and deterministic, but not guaranteed to reproduce
      all edge-case behaviors of the canonical implementation.
    - Intended for advanced workflows and controlled experimentation.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

MANIFEST = {
    "name": "JLC ControlNet Orchestrator (Advanced)",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": (
        "Extended version of the JLC ControlNet Orchestrator with integrated ControlNet loading "
        "and persistent caching. Implements the same non-canonical, non-recursive execution model "
        "using slot-based independent ControlNet evaluation and deterministic weighted fusion. "
        "Eliminates the need for chained ControlNet Apply nodes and external loader nodes while "
        "preserving correct conditioning injection. Maintains slot-order invariance (alpha=1), "
        "prevents cross-contamination via copy-based isolation, and safely falls back to native "
        "ControlNet Apply semantics for single-ControlNet cases. Built-in loaders and cache reduce "
        "redundant model loading while maintaining strict execution correctness."
    ),
}


import torch

DEBUG = True

import folder_paths
import comfy.controlnet

# ------------------------------------------------------------
# 🔒 Global ControlNet Cache (max 3 expected, no eviction)
# ------------------------------------------------------------
GLOBAL_CONTROLNET_CACHE = {}

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
class JLC_ControlNetOrchestratorAdvanced:

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
                "strength_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_01": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "control_net_name_01": (
                    ["DISABLED"] + folder_paths.get_filename_list("controlnet"),
                    {
                        "tooltip": "ControlNet selector"
                    }
                ),

                # --- SLOT 2 ---
                "image_02": ("IMAGE",),
                "strength_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_02": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "control_net_name_02": (
                    ["DISABLED", "SHARE_PREVIOUS"] + folder_paths.get_filename_list("controlnet"),
                    {
                        "tooltip": "ControlNet selector"
                    }
                ),

                # --- SLOT 3 ---
                "image_03": ("IMAGE",),
                "strength_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_03": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "control_net_name_03": (
                    ["DISABLED", "SHARE_PREVIOUS"] + folder_paths.get_filename_list("controlnet"),
                    {
                        "tooltip": "ControlNet selector"
                    }
                ),

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

        alpha = kwargs.get("alpha", 1.0)
        
        # Checking only the incoming cnet
        current_base = None

        for i in range(1, 4):
            idx = f"{i:02d}"
            image = kwargs.get(f"image_{idx}")
            strength = kwargs.get(f"strength_{idx}", 1.0)
            start = kwargs.get(f"start_{idx}", 0.0)
            end = kwargs.get(f"end_{idx}", 1.0)
            weight = kwargs.get(f"weight_{idx}", 1.0)
            name = kwargs.get(f"control_net_name_{idx}")

            # ----------------------------------------
            # HARD BYPASS
            # ----------------------------------------
            if not name or name == "DISABLED":
                continue

            # ----------------------------------------
            # SHARE PREVIOUS
            # ----------------------------------------
            if name == "SHARE_PREVIOUS":
                if current_base is None:
                    continue
                candidate_base = current_base

            # ----------------------------------------
            # LOAD / CACHE
            # ----------------------------------------
            else:
                path = folder_paths.get_full_path_or_raise("controlnet", name)
                if path not in GLOBAL_CONTROLNET_CACHE:
                    GLOBAL_CONTROLNET_CACHE[path] = comfy.controlnet.load_controlnet(path)
                candidate_base = GLOBAL_CONTROLNET_CACHE[path]

            # ------------------------------------------------------------
            # 🚫 EARLY BYPASS (CRITICAL)
            # ------------------------------------------------------------
            if (
                image is None
                or strength == 0
                or weight == 0
                or (end - start) <= 0
            ):
                continue
            
            current_base = candidate_base

            resolved.append({
                "slot": i,
                "base": candidate_base,
                "image": image,
                "strength": strength,
                "start": start,
                "end": end,
                "weight": weight,
            })

        active = [str(item["slot"]) for item in resolved]
        inactive = [str(i) for i in range(1, 4) if str(i) not in active]

        print(f"[JLC-Orchestrator] Active: {', '.join(active) or 'none'} | Inactive: {', '.join(inactive)}")

        if not resolved:
            return (positive, negative)

        # ------------------------------------------------------------
        # Phase 2 — Build independent ControlNets
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