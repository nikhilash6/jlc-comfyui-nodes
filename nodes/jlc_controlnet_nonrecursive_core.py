"""
JLC ControlNet Non-Recursive ControlNet Composition Core
---------------------------------

- JLC ComfyUI Nodes Collection
  - This helper is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

- Helper Purpose
  - This module contains the shared non-recursive ControlNet composition
    primitives used by the JLC ControlNet Composition and Orchestrator nodes.

  - The core design principle is that the sampler should see exactly one
    ControlNet-like object.  Internally, that wrapper evaluates detached
    ControlNets independently and combines their outputs by weighted additive
    streaming accumulation.

  - The shared helper provides:
        • extraction of native `previous_controlnet` chains in oldest-to-newest
          order;
        • shallow detachment of chain elements with `copy.copy(...)`;
        • explicit reset of `previous_controlnet` on detached children;
        • current ComfyUI compatibility shunts such as `multigpu_clones = {}`,
          `get_models_only_self()`, and `get_instance_for_device(...)`;
        • native single-ControlNet routing when fusion is unnecessary;
        • debug logging for chain order, weights, alpha, active slots, and
          native-versus-composed decisions.

  - The helper deliberately does not own model loading, cache policy, VAE
    behavior, preprocessors, or workflow-slot UI state.  Those responsibilities
    remain with the node classes that call into this core.

  - Algorithmic invariants:
        • no `copy.deepcopy()`;
        • no real MultiGPU clone implementation;
        • no mutation of upstream ControlNet chains;
        • no change to ControlNet math or chain order;
        • single active ControlNet uses native
          `set_previous_controlnet(prev_cnet)`;
        • two or more active ControlNets use the JLC non-recursive composed
          wrapper.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in the core **ComfyUI**
    project:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

import copy
import torch

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_HELPERS_VERSION 

MANIFEST = {
    "name": "JLC ControlNet Non-Recursive ControlNet Composition Core",
    "version": JLC_CONTROLNET_HELPERS_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Shared non-recursive ControlNet composition helper. Provides native "
        "chain extraction, shallow detachment, a sampler-facing composed "
        "ControlNet wrapper, compatibility shunts for current ComfyUI, and "
        "native single-ControlNet routing without owning model loading or cache "
        "policy."
    ),
}

DEBUG = True


def _log(message):
    if DEBUG:
        print(message)


def clear_multigpu_clone_state(controlnet):
    """
    Clear inherited ComfyUI MultiGPU clone bookkeeping from an isolated
    ControlNet-like object.

    JLC non-recursive composition intentionally behaves as a single sampler-
    facing object and does not participate in real MultiGPU cloning.
    """
    if controlnet is not None and hasattr(controlnet, "multigpu_clones"):
        controlnet.multigpu_clones = {}
    return controlnet


# Backward-friendly private alias for node code that historically used this
# exact helper name locally.
_clear_multigpu_clone_state = clear_multigpu_clone_state


def safe_cnet_name(controlnet):
    model = getattr(controlnet, "control_model", None)
    if model is not None:
        return model.__class__.__name__
    if controlnet is None:
        return "None"
    return controlnet.__class__.__name__


_safe_cnet_name = safe_cnet_name


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
        clear_multigpu_clone_state(c_copy)
        detached.append(c_copy)

    return detached


# ------------------------------------------------------------
# Wrapper: behaves like ONE ControlNet to the sampler
# ------------------------------------------------------------
class JLC_ComposedControlNet:
    def __init__(self, controlnets, weights, debug_label="JLC-ControlNet", debug_names=None, debug_alpha=None):
        self.controlnets = controlnets
        self.weights = weights
        self.previous_controlnet = None
        self.extra_hooks = None

        # Current ComfyUI compatibility: ControlNet-like objects are expected
        # to expose a MultiGPU clone registry. Empty dict means explicit
        # single-GPU/no-clone behavior for this wrapper.
        self.multigpu_clones = {}

        self._jlc_debug_label = debug_label
        self._jlc_debug_names = debug_names
        self._jlc_debug_alpha = debug_alpha
        self._jlc_debug_first_get_control = True

        if DEBUG:
            names = debug_names if debug_names is not None else [safe_cnet_name(c) for c in controlnets]
            _log(
                f"[{debug_label}] Compose wrapper created "
                f"chain_order={names} weights={weights} alpha={debug_alpha}"
            )

    # ---------------------------------------------------------------------
    # Core April-style algorithm: evaluate detached ControlNets independently
    # and fuse their outputs by weighted additive streaming accumulation.
    # ---------------------------------------------------------------------
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        combined = None
        first_debug = DEBUG and self._jlc_debug_first_get_control
        self._jlc_debug_first_get_control = False

        for index, (cnet, w) in enumerate(zip(self.controlnets, self.weights), start=1):
            if cnet is None or w == 0:
                if first_debug:
                    _log(
                        f"[{self._jlc_debug_label}] Child {index} skipped "
                        f"name={safe_cnet_name(cnet)} weight={w}"
                    )
                continue

            if first_debug:
                name = (
                    self._jlc_debug_names[index - 1]
                    if self._jlc_debug_names is not None and index - 1 < len(self._jlc_debug_names)
                    else safe_cnet_name(cnet)
                )
                _log(
                    f"[{self._jlc_debug_label}] Child {index} first_eval "
                    f"name={name} weight={w}"
                )

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


def _inject_single_native(conditioning, single_cnet, cnets):
    out = []

    for t in conditioning:
        d = t[1].copy()
        prev_cnet = d.get("control", None)

        if prev_cnet in cnets:
            c_net = cnets[prev_cnet]
        else:
            c_net = single_cnet.copy()
            clear_multigpu_clone_state(c_net)
            c_net.set_previous_controlnet(prev_cnet)
            cnets[prev_cnet] = c_net

        d["control"] = c_net
        d["control_apply_to_uncond"] = False
        out.append([t[0], d])

    return out


def _inject_composed(conditioning, composed):
    out = []
    for t in conditioning:
        d = t[1].copy()
        d["control"] = composed
        d["control_apply_to_uncond"] = False
        out.append([t[0], d])
    return out


def compose_or_native_fallback(
    positive,
    negative,
    controlnets,
    weights,
    alpha=1.0,
    debug_label="JLC-ControlNet",
    debug_names=None,
):
    """
    Shared Orchestrator fallback/compose decision.

    - No active ControlNets: pass through unchanged.
    - One active ControlNet: native Comfy chaining via set_previous_controlnet.
    - Multiple active ControlNets: one sampler-facing JLC_ComposedControlNet.

    This helper intentionally does not load models and does not interact with
    any model residency cache.
    """
    active_count = len(controlnets)

    if active_count == 0:
        if DEBUG:
            _log(f"[{debug_label}] Fallback decision=passthrough active_count=0")
        return (positive, negative)

    if active_count == 1:
        if DEBUG:
            names = debug_names if debug_names is not None else [safe_cnet_name(controlnets[0])]
            _log(
                f"[{debug_label}] Fallback decision=native_single "
                f"chain_order={names} raw_weights={weights} alpha={alpha}"
            )

        cnets = {}
        return (
            _inject_single_native(positive, controlnets[0], cnets),
            _inject_single_native(negative, controlnets[0], cnets),
        )

    final_weights = [w * (alpha ** i) for i, w in enumerate(weights)]

    if DEBUG:
        names = debug_names if debug_names is not None else [safe_cnet_name(c) for c in controlnets]
        _log(
            f"[{debug_label}] Fallback decision=composed "
            f"chain_order={names} raw_weights={weights} "
            f"final_weights={final_weights} alpha={alpha}"
        )

    composed = JLC_ComposedControlNet(
        controlnets,
        final_weights,
        debug_label=debug_label,
        debug_names=debug_names,
        debug_alpha=alpha,
    )
    return (_inject_composed(positive, composed), _inject_composed(negative, composed))
