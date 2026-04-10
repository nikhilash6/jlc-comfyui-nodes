"""
JLC ControlNet Composition

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
    - The **JLC ControlNet Composition** node replaces ComfyUI’s
      recursive ControlNet chaining with an explicit, non-recursive
      parallel composition model.

    - Instead of linked execution via `previous_controlnet`, this node:
            • Extracts the full ControlNet chain
            • Detaches it safely (no mutation of originals)
            • Evaluates each ControlNet independently
            • Combines outputs using weighted additive fusion

    - Composition is defined as:
            combined = Σ (w_i · α^i) · C_i(x)

      where:
            • w_i = user-defined weights (can be negative for experimentation)
            • α = order bias controlling dominance across ControlNets

    - This approach:
            • Eliminates recursive traversal overhead
            • Reduces memory pressure and improves scaling
            • Enables precise control over multi-ControlNet interactions
            • Preserves compatibility with ComfyUI conditioning pipelines

    - The node is particularly effective in:
            • multi-ControlNet workflows (3+ CNs)
            • high-resolution or tiled pipelines
            • LoRA + ControlNet combined setups

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

MANIFEST = {
    "name": "JLC ControlNet Composition",
    "version": (1, 1, 0),
    "author": "J. L. Córdova",
    "description": (
            "Node that implements a novel non-recursive ControlNet composition "
            "approach, replacing recursive chaining with explicit weighted fusion. "
            "Uses a deterministic streaming accumulation strategy with explicit GPU "
            "synchronization to control execution order, reduce peak memory pressure, "
            "and improve stability across multi-ControlNet workflows."
    ),
}

import copy
import torch

# ------------------------------------------------------------
# 🔧 Wrapper: behaves like ONE ControlNet to the sampler
# ------------------------------------------------------------
class JLC_ComposedControlNet:
    def __init__(self, controlnets, weights):
        self.controlnets = controlnets
        self.weights = weights
        self.previous_controlnet = None
        self.extra_hooks = None

    # ------------------------------------------------------------------------
    # This is the fundamental concept of this approach. Implements the ComfyUI ControlNet
    # interface (see comfy/controlnet.py). Conceptually derived from ControlNet.get_control,
    # but replaces recursive chaining and control_merge with explicit parallel
    # evaluation and weighted fusion.
    # ------------------------------------------------------------------------
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        combined = None

        for cnet, w in zip(self.controlnets, self.weights):
            # ------------------------------------------------------------
            # 🔵 Phase 1 — Select active ControlNet
            # ------------------------------------------------------------
            if cnet is None or w == 0:
                continue

            # ------------------------------------------------------------
            # 🔵 Phase 2 — Execute ControlNet (produce tensors)
            # ------------------------------------------------------------
            out = cnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

            if out is None:
                continue

            # ------------------------------------------------------------
            # 🟡 Phase 3 — Synchronization barrier (CRITICAL)
            # Ensures:
            #   • all GPU kernels for this ControlNet are finished
            #   • memory is actually materialized and stable
            #   • prevents overlap with next ControlNet execution
            # ------------------------------------------------------------
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # ------------------------------------------------------------
            # 🟢 Phase 4 — Ownership (FIRST CLONE ONLY)
            # Establish combined structure and take ownership of tensors
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
            # 🟣 Phase 5 — Streaming accumulation (in-place, no temps)
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

                    # local reference drop (helps Python GC timing)
                    del out_list

            # ------------------------------------------------------------
            # 🔴 Phase 6 — Immediate release of ControlNet output
            # ------------------------------------------------------------
            del out

            # (optional experimental hook — see notes below)
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

        # ------------------------------------------------------------
        # ⚫ Phase 7 — Final result
        # ------------------------------------------------------------
        return combined


    # --------------------------------------------------
    # REQUIRED for compatibility: hook aggregation
    # --------------------------------------------------
    def get_extra_hooks(self):
        hooks = []
        for cnet in self.controlnets:
            if hasattr(cnet, "get_extra_hooks"):
                hooks += cnet.get_extra_hooks()
        return hooks

    # --------------------------------------------------
    # REQUIRED for compatibility: model loading aggregation
    # --------------------------------------------------
    def get_models(self):
        models = []
        for cnet in self.controlnets:
            if hasattr(cnet, "get_models"):
                models += cnet.get_models()
        return models

    # --------------------------------------------------
    # REQUIRED for compatibility: estimates of memory requirements
    # considering only largest child ControlNet
    # --------------------------------------------------
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

    # --------------------------------------------------
    # REQUIRED for compatibility: pre-run (sets timestep ranges, etc.)
    # --------------------------------------------------
    def pre_run(self, model, percent_to_timestep_function):
        for cnet in self.controlnets:
            if hasattr(cnet, "pre_run"):
                cnet.pre_run(model, percent_to_timestep_function)

    # --------------------------------------------------
    # REQUIRED for compatibility: cleanup
    # --------------------------------------------------
    def cleanup(self):
        for cnet in self.controlnets:
            if hasattr(cnet, "cleanup"):
                cnet.cleanup()

        # aggressive reference break
        # self.controlnets = []


# ------------------------------------------------------------
# 🧠 Helper: extract full chain, oldest -> newest
# ------------------------------------------------------------
def extract_controlnet_chain(cnet):
    chain = []
    visited = set()

    while cnet is not None and id(cnet) not in visited:
        chain.append(cnet)
        visited.add(id(cnet))
        cnet = getattr(cnet, "previous_controlnet", None)

    chain.reverse()
    return chain


# ------------------------------------------------------------
# 🧠 Helper: detach chain (break recursion safely)
# ------------------------------------------------------------
def make_detached_chain(chain):
    detached = []

    for c in chain:
        c_copy = copy.copy(c)  # shallow copy is enough here
        if hasattr(c_copy, "previous_controlnet"):
            c_copy.previous_controlnet = None
        detached.append(c_copy)

    return detached


# ------------------------------------------------------------
# 🎯 Main Node
# ------------------------------------------------------------
class JLC_ControlNetComposition:
    FUNCTION = "compose_controlnet"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
            "optional": {
                "weight_1": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Contribution of ControlNet 1 (can be negative)"
                }),
                "weight_2": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Contribution of ControlNet 2 (can be negative)"
                }),
                "weight_3": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Contribution of ControlNet 3 (can be negative)"
                }),
                "weight_4": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Contribution of ControlNet 4 (can be negative)"
                }),
                "weight_5": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Contribution of ControlNet 5 (can be negative)"
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Order bias. <1 favors earlier ControlNets, >1 favors later ones"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")

    def compose_controlnet(
        self,
        positive,
        negative,
        weight_1=1.0,
        weight_2=1.0,
        weight_3=1.0,
        weight_4=1.0,
        weight_5=1.0,
        alpha=1.0,
    ):
        weights = [weight_1, weight_2, weight_3, weight_4, weight_5]

        def process_conditioning(conditioning):
            out = []

            for t in conditioning:
                d = t[1].copy()
                cnet = d.get("control", None)

                if cnet is None:
                    out.append([t[0], d])
                    continue

                # 🔍 Extract full chain
                chain = extract_controlnet_chain(cnet)

                # ✂️ Break recursion
                detached_chain = make_detached_chain(chain)

                # 🎚️ Trim / match weights
                trimmed_chain = detached_chain[:len(weights)]

                if not trimmed_chain:
                    out.append([t[0], d])
                    continue

                trimmed_weights = weights[:len(trimmed_chain)]

                # 🎯 Single composition law (decay-based weighting)
                final_weights = [
                    w * (alpha ** i)
                    for i, w in enumerate(trimmed_weights)
                ]

                print(f"[JLC-ControlNet] ⚙️ ControlNet composition using weights={final_weights} alpha={alpha}")

                # 🧩 Create composed wrapper
                composed = JLC_ComposedControlNet(
                    trimmed_chain,
                    final_weights,
                )

                # 🔁 Inject back
                d["control"] = composed
                out.append([t[0], d])

            return out

        new_positive = process_conditioning(positive)
        new_negative = process_conditioning(negative)

        return (new_positive, new_negative)