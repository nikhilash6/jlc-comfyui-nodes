"""
JLC ControlNet Composition
--------------------------

Maintenance-rescue version.

This file intentionally preserves the April-style non-recursive composition
algorithm.  The changes here are compatibility/state-hygiene shims only:

- keep the sampler-facing object as one ControlNet-like wrapper;
- expose an empty ``multigpu_clones`` mapping for current ComfyUI;
- provide lightweight current-interface helpers;
- clear inherited MultiGPU clone bookkeeping from detached shallow copies;
- keep chain detachment shallow via ``copy.copy``;
- do not use ``copy.deepcopy`` or real MultiGPU deep-cloning.
"""

MANIFEST = {
    "name": "JLC ControlNet Composition",
    "version": (1, 1, 1),
    "author": "J. L. Córdova",
    "description": (
        "Non-recursive ControlNet composition with current ComfyUI compatibility "
        "shims and dynamic visible weight rows."
    ),
}

import copy
import torch

MAX_SLOTS = 10
DEBUG = True


# ------------------------------------------------------------
# Small compatibility helpers
# ------------------------------------------------------------
def _clear_multigpu_clone_state(controlnet):
    """
    Modern ComfyUI ControlNet objects carry a multigpu_clones dict.

    JLC composition intentionally runs as a single sampler-facing wrapper with
    detached shallow child ControlNets.  A detached child should not inherit any
    alternate-device clone registry from the original object.
    """
    if hasattr(controlnet, "multigpu_clones"):
        controlnet.multigpu_clones = {}
    return controlnet


def _safe_cnet_name(controlnet):
    model = getattr(controlnet, "control_model", None)
    if model is not None:
        return model.__class__.__name__
    return controlnet.__class__.__name__


# ------------------------------------------------------------
# Wrapper: behaves like ONE ControlNet to the sampler
# ------------------------------------------------------------
class JLC_ComposedControlNet:
    def __init__(self, controlnets, weights):
        self.controlnets = controlnets
        self.weights = weights
        self.previous_controlnet = None
        self.extra_hooks = None

        # Current ComfyUI compatibility: ControlNet-like objects are expected
        # to expose a MultiGPU clone registry.  Empty dict means explicit
        # single-GPU/no-clone behavior for this wrapper.
        self.multigpu_clones = {}

    # ---------------------------------------------------------------------
    # Core April-style algorithm: evaluate detached ControlNets independently
    # and fuse their outputs by weighted additive streaming accumulation.
    # ---------------------------------------------------------------------
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        combined = None

        for cnet, w in zip(self.controlnets, self.weights):
            if cnet is None or w == 0:
                continue

            out = cnet.get_control(x_noisy, t, cond, batched_number, transformer_options)
            if out is None:
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()

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

            del out

        return combined

    # --------------------------------------------------
    # ControlNet-like compatibility surface
    # --------------------------------------------------
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

    def get_models_only_self(self):
        return self.get_models()

    def get_instance_for_device(self, device):
        return self

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
# Extract full chain, oldest -> newest
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
# Detach chain without recursive/deep copying
# ------------------------------------------------------------
def make_detached_chain(chain):
    detached = []

    for cnet in chain:
        c_copy = copy.copy(cnet)
        if hasattr(c_copy, "previous_controlnet"):
            c_copy.previous_controlnet = None
        _clear_multigpu_clone_state(c_copy)
        detached.append(c_copy)

    return detached


# ------------------------------------------------------------
# Main Node
# ------------------------------------------------------------
class JLC_ControlNetComposition:
    FUNCTION = "compose_controlnet"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "slot_count": ("INT", {
                "default": 5,
                "min": 1,
                "max": MAX_SLOTS,
                "step": 1,
                "tooltip": "Number of visible/active ControlNet weight rows. Backend ignores weights above this count.",
            }),
        }

        for i in range(1, MAX_SLOTS + 1):
            idx = f"{i:02d}"
            optional[f"weight_{idx}"] = ("FLOAT", {
                "default": 1.0,
                "min": -10.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": f"Contribution of ControlNet {i} in extracted chain; can be negative.",
            })

        optional["alpha"] = ("FLOAT", {
            "default": 1.0,
            "min": 0.01,
            "max": 2.0,
            "step": 0.01,
            "tooltip": "Order bias. <1 favors earlier ControlNets, >1 favors later ones.",
        })

        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")

    def compose_controlnet(self, positive, negative, slot_count=5, alpha=1.0, **kwargs):
        slot_count = max(1, min(MAX_SLOTS, int(slot_count)))
        weights = [kwargs.get(f"weight_{i:02d}", 1.0) for i in range(1, slot_count + 1)]

        def process_conditioning(conditioning):
            out = []

            for t in conditioning:
                d = t[1].copy()
                cnet = d.get("control", None)

                if cnet is None:
                    out.append([t[0], d])
                    continue

                chain = extract_controlnet_chain(cnet)

                # Single-ControlNet path: leave native chain behavior untouched.
                if len(chain) <= 1:
                    out.append([t[0], d])
                    continue

                detached_chain = make_detached_chain(chain)
                trimmed_chain = detached_chain[:slot_count]

                if len(trimmed_chain) <= 1:
                    out.append([t[0], d])
                    continue

                trimmed_weights = weights[:len(trimmed_chain)]
                final_weights = [w * (alpha ** i) for i, w in enumerate(trimmed_weights)]

                if DEBUG:
                    names = [_safe_cnet_name(c) for c in trimmed_chain]
                    print(f"[JLC-ControlNet] Composition chain={names} weights={final_weights} alpha={alpha}")

                d["control"] = JLC_ComposedControlNet(trimmed_chain, final_weights)
                out.append([t[0], d])

            return out

        return (process_conditioning(positive), process_conditioning(negative))
