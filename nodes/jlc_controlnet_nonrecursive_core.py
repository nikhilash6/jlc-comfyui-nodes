"""
JLC ControlNet Non-Recursive ControlNet Composition Core
--------------------------------------------------------

- JLC ComfyUI Nodes Collection
  - This internal module is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for
    image-generation pipelines, particularly:
        • Flux-based workflows
        • LoRA experimentation
        • advanced inpainting / outpainting pipelines
        • multi-ControlNet composition and orchestration

- Module Purpose
  - The **JLC ControlNet Non-Recursive ControlNet Composition Core**
    is the shared mathematical and compatibility layer used by:
        • JLC ControlNet Composition
        • JLC ControlNet Orchestrator
        • JLC ControlNet Orchestrator (Advanced)

  - This module is not a ComfyUI node. It contains the common primitives
    that define how JLC ControlNet chains are extracted, isolated, prepared,
    routed, evaluated, and fused.

  - Centralizing these operations ensures that the modular Composition
    workflow and both Orchestrator variants use the same validated
    non-recursive execution model rather than maintaining independent
    implementations of the composition algorithm.

- Prime Architectural Invariants
  - The shared implementation preserves the following invariants:
        • no use of `deepcopy`
        • no mutation of upstream native ControlNet chains
        • no mutation of cached raw ControlNet base objects
        • shallow isolation of child ControlNets where required
        • removal of recursive linkage before composed execution
        • independent evaluation of each prepared ControlNet
        • explicit tensor ownership before in-place accumulation
        • one sampler-facing ControlNet-compatible composition object
        • native routing whenever it is mathematically equivalent
        • no implicit or partial implementation of real MultiGPU support

- Mathematical Model
  - The core implements linear weighted ControlNet composition:

        combined(x) = Σ [W_i · C_i(x)]

    where:
        • C_i(x) is the output of child ControlNet `i`, evaluated
          independently against the same sampler state
        • W_i is the final effective weight supplied to the composed wrapper
        • child outputs interact only through additive output aggregation

  - For callers that expose an order-bias parameter, the effective weight is:

        W_i = w_i · alpha^i

    where:
        • w_i is the user-defined ControlNet weight
        • alpha is the caller-defined order-bias parameter
        • i is the original zero-based chain or slot position
        • skipped or zero-weight slots do not renumber later exponents
        • negative weights are supported
        • negative alpha values are supported when permitted by the caller

  - The composed wrapper itself receives the final effective weights.
    It does not reinterpret, normalize, or redistribute them.

  - This is a linearized non-recursive model. Child ControlNets are treated
    as independent operators whose interaction occurs only when their output
    tensors are combined.

- Native Chain Extraction
  - `extract_controlnet_chain()` reads an existing ComfyUI
    `previous_controlnet` chain.

  - The extracted chain is returned in oldest-to-newest order so that visible
    weights and order bias correspond to the workflow's original ControlNet
    construction order.

  - Object identity is tracked during traversal to prevent malformed cyclic
    chains from causing infinite recursion or unbounded traversal.

- Shallow Detachment
  - `make_detached_chain()` isolates extracted ControlNets using
    `copy.copy()`.

  - For every detached child:
        • the original upstream object remains unchanged
        • `previous_controlnet` is set to `None`
        • inherited `multigpu_clones` bookkeeping is cleared
        • underlying model patchers and model weights remain shared
        • expensive or unsafe deep copying is avoided

  - Shallow detachment removes recursive execution linkage while preserving
    the prepared ControlNet state needed for inference.

- Per-Run ControlNet Preparation
  - `prepare_controlnet_copy()` provides the shared native-style preparation
    path used by the Orchestrator nodes.

  - It performs:

        base_controlnet.copy().set_cond_hint(
            control_hint,
            strength,
            timestep_percent_range,
            vae=vae,
        )

  - This ensures that:
        • cached or externally supplied base objects are not conditioned
          in-place
        • hint state belongs to an isolated per-run ControlNet object
        • native ComfyUI hint, strength, activation-range, and VAE handling
          remain in effect
        • stale MultiGPU clone bookkeeping is removed from the prepared copy

- Sampler-Facing Composed Wrapper
  - `JLC_ComposedControlNet` presents the prepared children as one
    ControlNet-compatible object to the ComfyUI sampler.

  - The wrapper stores:
        • the ordered child ControlNet objects
        • their final effective weights
        • `previous_controlnet = None`
        • an empty `multigpu_clones` compatibility dictionary
        • optional diagnostic names and metadata

  - The wrapper does not construct a hidden native ControlNet chain and does
    not merge or rewrite child model weights.

- Streaming Fusion and Tensor Ownership
  - During `get_control()`, each non-zero child is evaluated independently:

        output_i = child_i.get_control(...)

  - The first available output tensor establishes ownership of the combined
    result:
        • the tensor is cloned
        • its effective weight is applied in-place when not equal to 1.0
        • the original child output tensor is never reused as mutable
          combined storage

  - Later outputs are accumulated into the owned destination tensor with:

        dst.add_(value, alpha=weight)

  - This strategy:
        • avoids unnecessary cloning of every child output
        • avoids mutating storage owned by child ControlNets
        • keeps accumulation explicit and deterministic
        • reduces avoidable temporary tensor allocation
        • supports positive, zero, and negative effective weights

  - The fusion logic also handles:
        • child outputs that return `None`
        • individual `None` tensors within output lists
        • output keys first introduced by later children
        • output lists of differing lengths
        • zero-weight or missing child ControlNets

- Synchronization Policy
  - Forced CUDA synchronization after every child was useful during early
    ownership and release investigations, but it can impose a significant
    performance penalty in current ComfyUI and DynamicVRAM execution.

  - Per-child synchronization is therefore disabled by default.

  - It remains available as an opt-in diagnostic or compatibility control
    through:

        JLC_CONTROLNET_SYNC_AFTER_CHILD=1

  - Enabling this setting changes execution synchronization behavior but does
    not change the composition mathematics.

- Native and Composed Routing
  - `compose_or_native_fallback()` provides the shared routing policy used
    by the Orchestrator nodes.

  - Routing behavior is:
        • no effective active ControlNets:
              return conditioning unchanged

        • one effective ControlNet at final weight 1.0:
              use the native ComfyUI ControlNet path

        • one ControlNet at a non-unit final weight:
              use the composed wrapper so the requested weight is honored

        • multiple effective ControlNets:
              use the composed wrapper and weighted non-recursive fusion

  - The native single-ControlNet path preserves any ControlNet already
    attached to the incoming conditioning by reconnecting it through
    `set_previous_controlnet()`.

  - Positive and negative conditioning rows reuse prepared native copies
    when they share the same previous-ControlNet identity.

- Conditioning Injection
  - Conditioning dictionaries are shallow-copied before ControlNet
    replacement.

  - The shared injection helpers set:

        control_apply_to_uncond = False

  - Composed execution installs the same sampler-facing wrapper into matching
    conditioning paths rather than creating unnecessary duplicate wrappers.

- ComfyUI Compatibility Surface
  - `JLC_ComposedControlNet` exposes the interfaces required by current
    ComfyUI sampler and model-management paths, including:
        • `get_control`
        • `get_models`
        • `get_models_only_self`
        • `get_extra_hooks`
        • `get_instance_for_device`
        • `inference_memory_requirements`
        • `pre_run`
        • `cleanup`
        • `copy`

  - `get_models()` aggregates child model objects and removes duplicates by
    object identity so ComfyUI can manage their loading and residency.

  - `get_extra_hooks()` forwards and combines child ControlNet hooks.

  - `pre_run()` forwards sampler preparation to each child.

  - `cleanup()` forwards cleanup once per unique child object.

  - `copy()` creates a shallow wrapper copy while retaining the already
    prepared child ControlNets. It does not create a new native chain or
    duplicate model ownership.

- Inference Memory Reporting
  - Child model weights are exposed through `get_models()` and remain under
    ComfyUI model-management control.

  - Temporary inference-memory requirements are reported as the largest
    child requirement rather than the sum of all child requirements because
    children are evaluated sequentially.

  - This distinction avoids representing sequential scratch requirements as
    though every child required its full temporary inference workspace
    simultaneously.

- Model-Management Scope
  - This module does not load ControlNet files and does not own shared-cache
    capacity or eviction policy.

  - Model sourcing remains the responsibility of:
        • upstream ComfyUI or third-party loaders
        • JLC ControlNet Apply Advanced
        • JLC ControlNet Orchestrator (Advanced)
        • the shared JLC model-cache layer

  - The core is designed to cooperate with normal ComfyUI loading,
    offloading, weight patching, and DynamicVRAM behavior by exposing child
    models and lifecycle information through the expected interfaces.

  - It does not replace ComfyUI device-residency policy and does not attempt
    to force models to remain loaded.

- MultiGPU Scope
  - `multigpu_clones = {}` is an explicit compatibility shunt, not an
    implementation of MultiGPU ControlNet execution.

  - `get_instance_for_device()` returns the current wrapper for ordinary
    single-device compatibility.

  - `deepclone_multigpu()` raises a clear runtime error rather than silently
    creating unsafe cross-device ownership or returning an invalid clone.

  - Real MultiGPU ControlNet cloning must not be inferred from the presence
    of the compatibility attributes and should only be implemented as a
    separate, explicitly designed feature.

- Diagnostics
  - Shared core diagnostics are quiet by default.

  - They can be enabled with:

        JLC_CONTROLNET_DEBUG=1

  - Diagnostic output may include:
        • wrapper creation and child order
        • final effective weights
        • native, composed, or passthrough routing
        • first child evaluation
        • skipped zero-weight children
        • optional synchronization state

  - Debugging changes observability only; it does not change the mathematical
    result.

- Role Within the JLC ControlNet Family
  - This module is the unifying implementation contract for the JLC
    non-recursive ControlNet family.

  - **JLC ControlNet Composition** supplies the core with ControlNets
    extracted from an existing native chain.

  - **JLC ControlNet Orchestrator** supplies externally loaded ControlNet
    objects and per-slot conditioning.

  - **JLC ControlNet Orchestrator (Advanced)** supplies internally loaded or
    shared cached models and per-slot conditioning.

  - The nodes differ in model sourcing and workflow construction, but their
    composed execution relies on this same weighted-fusion implementation.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by and interoperable with the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
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
        "Shared mathematical and compatibility core for the JLC non-recursive "
        "ControlNet family. Provides cycle-safe native-chain extraction, "
        "shallow detachment without upstream mutation, isolated per-run "
        "ControlNet preparation, native single-ControlNet routing, and a "
        "sampler-facing composed wrapper implementing explicit streaming "
        "weighted fusion with first-output ownership and later in-place "
        "accumulation. Exposes current ComfyUI model, hook, lifecycle, "
        "inference-memory, and single-device compatibility interfaces while "
        "keeping real MultiGPU cloning explicitly unsupported."
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
