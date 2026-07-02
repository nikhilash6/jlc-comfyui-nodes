"""
JLC ControlNet Composition
--------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for
    image-generation pipelines, particularly:
        • Flux-based workflows
        • LoRA experimentation
        • advanced inpainting / outpainting pipelines
        • multi-ControlNet composition and orchestration

- Node Purpose
  - The **JLC ControlNet Composition** node is the original cornerstone
    of the JLC linearized non-recursive ControlNet architecture.

  - It converts an already-built native ComfyUI ControlNet chain into
    an explicit weighted composition that presents one ControlNet-compatible
    object to the sampler.

  - The typical modular workflow is:

        Apply Advanced → Apply Advanced → ... → Composition

    In this workflow:
        • Apply Advanced nodes construct an ordinary native
          `previous_controlnet` chain
        • Composition extracts the chain before sampling
        • each ControlNet is detached from recursive linkage
        • the detached ControlNets are evaluated independently
        • their outputs are combined through weighted additive fusion

  - This modular workflow is a first-class counterpart to
    **JLC ControlNet Orchestrator (Advanced)**.

    Both paths use the same validated non-recursive fusion core:
        • Composition receives a chain assembled elsewhere
        • Orchestrator Advanced prepares its ControlNets internally
        • the sampler-facing execution and fusion mathematics are shared

- Chain Extraction and Isolation
  - Composition reads the native `previous_controlnet` chain in its
    original oldest-to-newest order.

  - Each selected ControlNet is isolated using `copy.copy()`:
        • upstream ControlNet objects are not mutated
        • `previous_controlnet` is set to `None` on each copy
        • inherited MultiGPU clone bookkeeping is cleared
        • underlying model patchers and model weights remain shared
        • `deepcopy` is never used

  - The `slot_count` setting determines how many ControlNets from the
    extracted chain participate in composition.

    If the extracted chain is longer than `slot_count`, later members are
    intentionally excluded. If the visible weight rows exceed the chain
    length, unused non-zero rows are reported through diagnostic warnings.

- Composition Mathematics
  - Multi-ControlNet composition is defined as:

        combined = Σ [(w_i · alpha^i) · C_i(x)]

    where:
        • C_i(x) is the output of ControlNet `i`, evaluated independently
          against the same sampler state
        • w_i is the user-defined weight for ControlNet `i`
        • alpha^i applies optional order bias across the extracted chain
        • chain indexing begins with exponent zero for the oldest ControlNet
        • negative ControlNet weights are supported
        • alpha values below 1 favor earlier ControlNets
        • alpha values above 1 favor later ControlNets

  - Composition changes how ControlNet outputs are aggregated. It does not
    alter each ControlNet's own hint, strength, activation range, VAE
    preparation, or internal inference behavior.

- Fusion and Tensor Ownership
  - Each detached ControlNet is evaluated independently rather than invoking
    the next ControlNet recursively through `previous_controlnet`.

  - The shared fusion core uses a streaming accumulation strategy:
        • the first available output tensor is cloned when ownership is taken
        • its effective weight is applied in-place
        • later ControlNet outputs are accumulated with:

              dst.add_(value, alpha=weight)

        • output tensors are not cloned unnecessarily
        • upstream ControlNet output storage is not mutated

  - Optional CUDA synchronization after each child evaluation is available
    only as a diagnostic or compatibility setting. It is disabled by default
    because current ComfyUI and DynamicVRAM execution generally perform better
    without forced per-child synchronization.

- Native Equivalent Paths
  - Composition avoids creating a wrapper when native execution is already
    mathematically equivalent.

  - If the original conditioning contains only one ControlNet at unit weight,
    the native ControlNet is left unchanged.

  - If chain clipping leaves one detached ControlNet at a final weight of 1.0,
    Composition uses that detached native ControlNet directly.

  - A single ControlNet with a non-unit effective weight still uses the
    composed wrapper so that the declared weighting remains exact.

- Conditioning Integration
  - Positive and negative conditioning metadata are shallow-copied before
    replacement.

  - Matching conditioning rows reuse the same replacement ControlNet object
    when their extracted chain and effective weights are identical. This
    avoids needless wrapper duplication and preserves sampler batching
    opportunities.

  - The node sets:

        control_apply_to_uncond = False

    in accordance with the surrounding ComfyUI ControlNet conditioning path.

- Runtime Integration
  - The composed wrapper exposes the interfaces expected by current ComfyUI
    sampler and model-management paths, including:
        • `previous_controlnet`
        • `multigpu_clones`
        • `get_control`
        • `get_models`
        • `get_extra_hooks`
        • `inference_memory_requirements`
        • `pre_run`
        • `cleanup`

  - Child model objects are exposed to ComfyUI model management while
    temporary inference-memory requirements are estimated from the largest
    sequential child requirement rather than by blindly summing all children.

  - Composition does not load ControlNet models and does not own model-cache
    policy. Model loading, reuse, and residency remain the responsibility of
    upstream loaders, Apply Advanced nodes, and ComfyUI model management.

  - The node is designed to cooperate with normal ComfyUI VRAM management
    and DynamicVRAM behavior. It does not replace ComfyUI loading, offloading,
    weight patching, or device-residency policy.

- Diagnostic Behavior
  - When JLC ControlNet debugging is enabled, the node reports:
        • extracted chain order
        • visible and effective weights
        • native or composed routing decisions
        • chain-length and slot-count mismatches
        • ignored or unused weight rows

  - These diagnostics are intended to make modular multi-ControlNet workflows
    inspectable without changing their mathematical behavior.

- Relationship to JLC Orchestrators
  - **JLC ControlNet Composition** is the modular composition interface.

  - **JLC ControlNet Orchestrator (Advanced)** is the integrated
    internal-loader interface.

  - **JLC ControlNet Orchestrator** is the direct external-input interface.

  - All three rely on the same validated linearized non-recursive
    weighted-fusion algorithm. Their primary differences concern workflow
    construction, model sourcing, and user interface—not composition math.

- MultiGPU Scope
  - The composed wrapper includes compatibility shunts required by current
    ComfyUI interfaces but does not implement real MultiGPU ControlNet
    cloning.

  - Use this node as a single-device composition path unless explicit
    MultiGPU support is added in a future implementation.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by and interoperable with the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

import math

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION
from .jlc_controlnet_nonrecursive_core import (
    DEBUG,
    JLC_ComposedControlNet,
    extract_controlnet_chain,
    make_detached_chain,
    safe_cnet_name,
)


MANIFEST = {
    "name": "JLC ControlNet Composition",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Original modular interface for the validated JLC non-recursive "
        "ControlNet composition architecture. Extracts native "
        "previous_controlnet chains, shallow-detaches their members without "
        "mutating upstream objects, applies chain clipping and order-biased "
        "weights, and evaluates selected ControlNets independently through "
        "the shared streaming weighted-fusion core. Preserves mathematically "
        "equivalent native single-ControlNet paths and integrates with current "
        "ComfyUI sampler, lifecycle, and model-management interfaces."
    ),
}


MAX_SLOTS = 8

def _is_one(value: float) -> bool:
    try:
        return math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-12)
    except Exception:
        return value == 1.0


def _is_zero(value: float) -> bool:
    try:
        return math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1e-12)
    except Exception:
        return value == 0


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
        nonzero_visible_weight_rows = [i + 1 for i, w in enumerate(weights) if not _is_zero(w)]
        warned = set()
        replacement_cache = {}

        if DEBUG:
            print(
                f"[JLC-ControlNet Composition] slot_count={slot_count} "
                f"raw_weights={weights} alpha={alpha}"
            )

        def warn_once(key, message):
            if not DEBUG or key in warned:
                return
            warned.add(key)
            print(f"[JLC-ControlNet Composition WARNING] {message}")

        def warn_for_chain_length(chain_length, stream_name, row_index, chain_names):
            if chain_length == 0:
                if nonzero_visible_weight_rows:
                    warn_once(
                        ("no_control", tuple(nonzero_visible_weight_rows)),
                        (
                            f"{stream_name}[{row_index}] has no upstream ControlNet, but "
                            f"visible non-zero weight rows are set: {nonzero_visible_weight_rows}. "
                            "Check upstream Apply/Orchestrator nodes, disabled toggles, "
                            "strength=0, or missing hint image connections."
                        ),
                    )
                return

            if chain_length < slot_count:
                unused_nonzero = [
                    row for row in range(chain_length + 1, slot_count + 1)
                    if not _is_zero(weights[row - 1])
                ]
                if unused_nonzero:
                    warn_once(
                        ("short_chain", chain_length, tuple(unused_nonzero)),
                        (
                            f"slot_count={slot_count} but extracted chain has only "
                            f"{chain_length} ControlNet(s). Unused non-zero weight rows: "
                            f"{unused_nonzero}. Extracted chain={chain_names}."
                        ),
                    )

            if chain_length > slot_count:
                ignored = list(range(slot_count + 1, chain_length + 1))
                warn_once(
                    ("long_chain", chain_length, slot_count),
                    (
                        f"extracted chain has {chain_length} ControlNet(s), but "
                        f"slot_count={slot_count}. Upstream ControlNet row(s) {ignored} "
                        f"will be intentionally ignored by Composition. Extracted chain={chain_names}."
                    ),
                )

        def build_replacement(chain, chain_names):
            trimmed_original = chain[:slot_count]
            trimmed_weights = weights[:len(trimmed_original)]
            final_weights = [w * (alpha ** i) for i, w in enumerate(trimmed_weights)]
            trimmed_names = chain_names[:len(trimmed_original)]

            # Key on original object identity, not detached copies, so matching
            # positive/negative rows can share the same replacement wrapper.
            cache_key = (
                tuple(id(c) for c in trimmed_original),
                tuple(final_weights),
                alpha,
                slot_count,
            )
            if cache_key in replacement_cache:
                return replacement_cache[cache_key]

            detached_chain = make_detached_chain(trimmed_original)

            # If trimming and weighting leave exactly one child at weight 1.0,
            # the mathematically equivalent fastest path is a detached native
            # single ControlNet. This also correctly honors chain clipping.
            if len(detached_chain) == 1 and _is_one(final_weights[0]):
                replacement = detached_chain[0]
                replacement_kind = "detached_native_single"
            else:
                replacement = JLC_ComposedControlNet(
                    detached_chain,
                    final_weights,
                    debug_label="JLC-ControlNet Composition",
                    debug_names=trimmed_names,
                    debug_alpha=alpha,
                )
                replacement_kind = "composed"

            replacement_cache[cache_key] = (replacement, replacement_kind, trimmed_names, trimmed_weights, final_weights)
            return replacement_cache[cache_key]

        def process_conditioning(conditioning, stream_name):
            out = []

            for row_index, t in enumerate(conditioning):
                d = t[1].copy()
                cnet = d.get("control", None)

                if cnet is None:
                    warn_for_chain_length(0, stream_name, row_index, [])
                    if DEBUG:
                        print(f"[JLC-ControlNet Composition] {stream_name}[{row_index}] fallback=no_control")
                    out.append([t[0], d])
                    continue

                chain = extract_controlnet_chain(cnet)
                chain_names = [safe_cnet_name(c) for c in chain]
                warn_for_chain_length(len(chain), stream_name, row_index, chain_names)

                trimmed_original = chain[:slot_count]
                if len(trimmed_original) == 0:
                    out.append([t[0], d])
                    continue

                # Original single chain at unit weight is already native and has
                # no recursion to rescue. Leave it untouched for speed/behavior.
                if len(chain) == 1 and slot_count >= 1 and _is_one(weights[0]):
                    if DEBUG:
                        print(
                            f"[JLC-ControlNet Composition] {stream_name}[{row_index}] "
                            f"fallback=native_single chain_order={chain_names}"
                        )
                    out.append([t[0], d])
                    continue

                replacement, replacement_kind, trimmed_names, trimmed_weights, final_weights = build_replacement(chain, chain_names)

                if DEBUG:
                    print(
                        f"[JLC-ControlNet Composition] {stream_name}[{row_index}] "
                        f"fallback={replacement_kind} original_chain={chain_names} "
                        f"trimmed_chain={trimmed_names} raw_weights={trimmed_weights} "
                        f"final_weights={final_weights} alpha={alpha}"
                    )

                d["control"] = replacement
                d["control_apply_to_uncond"] = False
                out.append([t[0], d])

            return out

        return (
            process_conditioning(positive, "positive"),
            process_conditioning(negative, "negative"),
        )


NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetComposition": JLC_ControlNetComposition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetComposition": "JLC ControlNet Composition",
}
