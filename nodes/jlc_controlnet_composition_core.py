"""
JLC ControlNet Composition Core
-------------------------------

Shared non-recursive ControlNet composition helper for JLC ControlNet nodes.

This module centralizes the composed ControlNet wrapper used by:
    • JLC ControlNet Composition
    • JLC Dynamic ControlNet Orchestrator - Advanced

Maintenance rule:
    The fusion algorithm is intentionally unchanged. This file only prevents
    the standalone Composition node and Advanced Orchestrator from drifting
    apart, and preserves the current best-effort ComfyUI ControlBase bridge
    hooks used by the Advanced Orchestrator.

The multi-GPU / device-clone compatibility surface is best-effort. It mirrors
ComfyUI's current ControlBase-facing attributes where possible, but it has not
been validated on an actual multi-GPU setup.

Released under the MIT License as part of the JLC ComfyUI Nodes Collection.
"""

from __future__ import annotations

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_HELPERS_VERSION

import copy
import torch


class JLC_ComposedControlNet:
    def __init__(self, controlnets, weights, *, _is_multigpu_clone=False):
        self.controlnets = controlnets
        self.weights = weights
        self.previous_controlnet = None
        self.extra_hooks = None

        # Current ComfyUI ControlBase compatibility.
        #
        # Newer sampler code expects every control object to expose a
        # ``multigpu_clones`` dictionary after/around ``pre_run``. Native
        # ControlNet objects provide this attribute. This composed wrapper is a
        # ControlNet-compatible façade, so it must provide the same interface.
        #
        # On normal single-GPU systems this remains an empty dict and is a no-op.
        # On multi-GPU systems, ``pre_run`` below builds per-device composed
        # façades from the underlying ControlNet device clones.
        self.multigpu_clones = {}
        self._is_multigpu_clone = bool(_is_multigpu_clone)

    def get_instance_for_device(self, device):
        """Return the per-device composed wrapper when ComfyUI asks for it."""
        return self.multigpu_clones.get(device, self)

    def set_previous_controlnet(self, previous_controlnet):
        """ControlBase compatibility; composition itself remains non-recursive."""
        self.previous_controlnet = previous_controlnet
        return self

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

        # Build composed per-device wrappers from underlying ControlNet clones.
        # This mirrors the current ComfyUI ControlBase interface and avoids the
        # sampler-side AttributeError on ``multigpu_clones``. It does not change
        # the fusion algorithm; it only exposes the composed wrapper on the same
        # device-specific interface expected by ComfyUI.
        if self._is_multigpu_clone:
            return

        device_keys = set()
        for cnet in self.controlnets:
            clones = getattr(cnet, "multigpu_clones", None)
            if clones:
                device_keys.update(clones.keys())

        self.multigpu_clones = {}
        for device in device_keys:
            device_controlnets = []
            for cnet in self.controlnets:
                if hasattr(cnet, "get_instance_for_device"):
                    device_controlnets.append(cnet.get_instance_for_device(device))
                else:
                    clones = getattr(cnet, "multigpu_clones", {}) or {}
                    device_controlnets.append(clones.get(device, cnet))

            self.multigpu_clones[device] = JLC_ComposedControlNet(
                device_controlnets,
                self.weights,
                _is_multigpu_clone=True,
            )


    def cleanup(self):
        for cnet in self.controlnets:
            if hasattr(cnet, "cleanup"):
                cnet.cleanup()

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
