"""
JLC ControlNet Non-Recursive ControlNet Composition Core
--------------------------------------------------------

Shared non-recursive ControlNet composition primitives for the JLC ControlNet
Composition and Orchestrator nodes.

Prime invariants:
    - no deepcopy;
    - no mutation of upstream native ControlNet chains;
    - child ControlNets are shallow-copied/detached where needed;
    - the sampler sees one ControlNet-like object for composed execution;
    - composed execution remains the April-style linear non-recursive fusion:
          combined = Σ w_i * C_i(x)
      with first-output ownership clone and later in-place accumulation;
    - MultiGPU remains an explicit no-clone shunt, not a real implementation.
"""

from __future__ import annotations

import copy
import math
import os
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
        "native single-ControlNet routing when mathematically equivalent."
    ),
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Quiet by default. Set JLC_CONTROLNET_DEBUG=1 for console diagnostics.
DEBUG = _env_bool("JLC_CONTROLNET_DEBUG", False)

# The April prototypes synchronized after each child ControlNet. That was useful
# while proving ownership/release behavior, but on modern Comfy/DynamicVRAM it can
# be a major performance cliff. Keep it as an opt-in diagnostic/stability shunt.
SYNC_AFTER_CHILD = _env_bool("JLC_CONTROLNET_SYNC_AFTER_CHILD", False)


def _log(message: str) -> None:
    if DEBUG:
        print(message)


def _is_effectively_one(value: float) -> bool:
    try:
        return math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-12)
    except Exception:
        return value == 1.0


def _is_effectively_zero(value: float) -> bool:
    try:
        return math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1e-12)
    except Exception:
        return value == 0


def _unique_by_id(items):
    seen = set()
    out = []
    for item in items:
        key = id(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def clear_multigpu_clone_state(controlnet):
    """
    Clear inherited ComfyUI MultiGPU clone bookkeeping from an isolated
    ControlNet-like object.

    JLC non-recursive composition intentionally remains a single-GPU/no-clone
    wrapper. This shunt prevents stale clone dictionaries from being inherited
    by shallow copies.
    """
    if controlnet is None:
        return controlnet
    try:
        controlnet.multigpu_clones = {}
    except Exception:
        pass
    return controlnet


# Backward-friendly private alias for older node code.
_clear_multigpu_clone_state = clear_multigpu_clone_state


def safe_cnet_name(controlnet):
    if controlnet is None:
        return "None"
    model = getattr(controlnet, "control_model", None)
    if model is not None:
        return model.__class__.__name__
    return controlnet.__class__.__name__


_safe_cnet_name = safe_cnet_name


def extract_controlnet_chain(cnet):
    """Extract a native previous_controlnet chain in oldest -> newest order."""
    chain = []
    visited = set()

    while cnet is not None and id(cnet) not in visited:
        chain.append(cnet)
        visited.add(id(cnet))
        cnet = getattr(cnet, "previous_controlnet", None)

    chain.reverse()
    return chain


def make_detached_chain(chain):
    """
    Shallow-copy and detach a native ControlNet chain.

    This deliberately uses copy.copy(...), not deepcopy. Child ControlNet model
    patchers remain shared; per-run mutable chain linkage is cleared.
    """
    detached = []

    for cnet in chain:
        c_copy = copy.copy(cnet)
        if hasattr(c_copy, "previous_controlnet"):
            c_copy.previous_controlnet = None
        clear_multigpu_clone_state(c_copy)
        detached.append(c_copy)

    return detached


def prepare_controlnet_copy(base_controlnet, control_hint, strength, timestep_percent_range, vae=None):
    """Native Comfy-style copy + set_cond_hint helper used by Orchestrators."""
    cnet = base_controlnet.copy().set_cond_hint(
        control_hint,
        strength,
        timestep_percent_range,
        vae=vae,
    )
    clear_multigpu_clone_state(cnet)
    return cnet


class JLC_ComposedControlNet:
    """Sampler-facing ControlNet-like wrapper for non-recursive composition."""

    def __init__(self, controlnets, weights, debug_label="JLC-ControlNet", debug_names=None, debug_alpha=None):
        self.controlnets = list(controlnets)
        self.weights = list(weights)
        self.previous_controlnet = None
        self.extra_hooks = None

        # Current ComfyUI compatibility shunt. Empty dict explicitly means that
        # this wrapper does not own/use real MultiGPU ControlNet clones.
        self.multigpu_clones = {}

        self._jlc_debug_label = debug_label
        self._jlc_debug_names = list(debug_names) if debug_names is not None else None
        self._jlc_debug_alpha = debug_alpha
        self._jlc_debug_first_get_control = True

        if DEBUG:
            names = self._jlc_debug_names or [safe_cnet_name(c) for c in self.controlnets]
            _log(
                f"[{debug_label}] composed wrapper created "
                f"chain_order={names} weights={self.weights} alpha={debug_alpha} "
                f"sync_after_child={SYNC_AFTER_CHILD}"
            )

    # ------------------------------------------------------------------
    # April-style core math. Do not replace with recursion, shared mutable
    # composition state, deepcopy, or hidden ControlNet merging.
    # ------------------------------------------------------------------
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        combined = None
        first_debug = DEBUG and self._jlc_debug_first_get_control
        self._jlc_debug_first_get_control = False

        for index, (cnet, w) in enumerate(zip(self.controlnets, self.weights), start=1):
            if cnet is None or _is_effectively_zero(w):
                if first_debug:
                    _log(
                        f"[{self._jlc_debug_label}] child {index} skipped "
                        f"name={safe_cnet_name(cnet)} weight={w}"
                    )
                continue

            if first_debug:
                name = (
                    self._jlc_debug_names[index - 1]
                    if self._jlc_debug_names is not None and index - 1 < len(self._jlc_debug_names)
                    else safe_cnet_name(cnet)
                )
                _log(f"[{self._jlc_debug_label}] child {index} first_eval name={name} weight={w}")

            out = cnet.get_control(x_noisy, t, cond, batched_number, transformer_options)
            if out is None:
                continue

            if SYNC_AFTER_CHILD and torch.cuda.is_available():
                torch.cuda.synchronize()

            if combined is None:
                combined = {}

                for key, out_list in out.items():
                    new_list = [None] * len(out_list)

                    for i, v in enumerate(out_list):
                        if v is None:
                            continue

                        owned = v.clone()
                        if not _is_effectively_one(w):
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
                            if not _is_effectively_one(w):
                                owned.mul_(w)
                            combined_list[i] = owned
                        else:
                            dst.add_(v, alpha=w)

                    del out_list

            del out

        return combined

    # ------------------------------------------------------------------
    # ControlNet-like compatibility surface for current Comfy sampler/model
    # management paths.
    # ------------------------------------------------------------------
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
        return _unique_by_id(models)

    def get_models_only_self(self):
        return self.get_models()

    def get_instance_for_device(self, device):
        return self

    def deepclone_multigpu(self, load_device, autoregister=False):
        """
        Explicit no-real-MultiGPU shunt.

        Returning self here would be unsafe because Comfy's MultiGPU path expects
        per-device ControlNet/model ownership. Raising a clear error is safer
        than silently mixing devices if MultiGPU is accidentally enabled.
        """
        raise RuntimeError(
            "JLC_ComposedControlNet does not implement real ComfyUI MultiGPU "
            "ControlNet cloning. Disable MultiGPU for JLC non-recursive "
            "ControlNet composition/orchestration."
        )

    def inference_memory_requirements(self, dtype):
        # Sequential child evaluation requires the largest child scratch budget,
        # not the sum. Child model weights are still exposed via get_models().
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
        for cnet in _unique_by_id(self.controlnets):
            if hasattr(cnet, "cleanup"):
                cnet.cleanup()

    def copy(self):
        # Shallow wrapper copy only. Children remain the already-prepared per-run
        # child controls; this is for interface compatibility, not a new chain.
        copied = JLC_ComposedControlNet(
            self.controlnets.copy(),
            self.weights.copy(),
            debug_label=self._jlc_debug_label,
            debug_names=self._jlc_debug_names.copy() if self._jlc_debug_names is not None else None,
            debug_alpha=self._jlc_debug_alpha,
        )
        return copied


def _inject_single_native(conditioning, single_cnet, cnets):
    out = []

    for t in conditioning:
        d = t[1].copy()
        prev_cnet = d.get("control", None)
        cache_key = id(prev_cnet)

        if cache_key in cnets:
            c_net = cnets[cache_key]
        else:
            c_net = single_cnet.copy()
            clear_multigpu_clone_state(c_net)
            c_net.set_previous_controlnet(prev_cnet)
            cnets[cache_key] = c_net

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


def _active_pairs(controlnets, weights, alpha):
    pairs = []
    for original_index, (cnet, raw_weight) in enumerate(zip(controlnets, weights)):
        final_weight = raw_weight * (alpha ** original_index)
        if cnet is None or _is_effectively_zero(final_weight):
            continue
        pairs.append((original_index, cnet, raw_weight, final_weight))
    return pairs


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
    Shared Orchestrator routing.

    - No effective active ControlNets: pass through unchanged.
    - One effective active ControlNet at final weight 1.0: native Comfy chain.
    - One effective active ControlNet at another weight, or multiple active
      ControlNets: JLC composed wrapper so the declared math is honored.

    This helper intentionally does not load models and does not own cache policy.
    """
    pairs = _active_pairs(controlnets, weights, alpha)

    if len(pairs) == 0:
        if DEBUG:
            _log(f"[{debug_label}] fallback=passthrough active_count=0")
        return (positive, negative)

    if len(pairs) == 1 and _is_effectively_one(pairs[0][3]):
        original_index, cnet, raw_weight, final_weight = pairs[0]
        if DEBUG:
            names = debug_names if debug_names is not None else [safe_cnet_name(cnet)]
            chosen_name = names[original_index] if original_index < len(names) else safe_cnet_name(cnet)
            _log(
                f"[{debug_label}] fallback=native_single name={chosen_name} "
                f"raw_weight={raw_weight} final_weight={final_weight} alpha={alpha}"
            )

        cnets = {}
        return (
            _inject_single_native(positive, cnet, cnets),
            _inject_single_native(negative, cnet, cnets),
        )

    filtered_controlnets = [p[1] for p in pairs]
    final_weights = [p[3] for p in pairs]
    if debug_names is not None:
        filtered_names = [debug_names[p[0]] if p[0] < len(debug_names) else safe_cnet_name(p[1]) for p in pairs]
    else:
        filtered_names = [safe_cnet_name(p[1]) for p in pairs]

    if DEBUG:
        _log(
            f"[{debug_label}] fallback=composed chain_order={filtered_names} "
            f"final_weights={final_weights} alpha={alpha}"
        )

    composed = JLC_ComposedControlNet(
        filtered_controlnets,
        final_weights,
        debug_label=debug_label,
        debug_names=filtered_names,
        debug_alpha=alpha,
    )
    return (_inject_composed(positive, composed), _inject_composed(negative, composed))
