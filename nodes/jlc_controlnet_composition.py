"""
JLC ControlNet Composition
--------------------------

Converts an already-built native ComfyUI ControlNet chain into an explicit
non-recursive weighted composition.

This pass keeps the April non-recursive fusion algorithm intact and only hardens
Comfy-facing behavior: imports are explicit, chain clipping is honored even when
it leaves one child, non-unit single weights are honored through the composed
wrapper, and identical conditioning rows share the same replacement wrapper so
sampler batching is not defeated by needless object duplication.
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
        "Converts an existing native ControlNet chain into a non-recursive "
        "weighted composition. Extracts and shallow-detaches upstream ControlNets, "
        "uses a sampler-facing composed wrapper for weighted fusion, preserves "
        "native behavior only when mathematically equivalent, and warns when "
        "visible weight rows do not match the extracted chain."
    ),
}

MAX_SLOTS = 10


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
