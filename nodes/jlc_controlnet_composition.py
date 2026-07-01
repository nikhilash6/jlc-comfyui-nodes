"""
JLC ControlNet Composition
--------------------------

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
        • multi-ControlNet composition and orchestration

- Node Purpose
  - The **JLC ControlNet Composition** node converts an already-built native
    ComfyUI ControlNet chain into an explicit non-recursive weighted
    composition.

  - Instead of allowing recursive evaluation through `previous_controlnet`,
    the node:
        • extracts the full upstream ControlNet chain;
        • preserves chain order from oldest to newest;
        • shallow-copies each ControlNet with `copy.copy(...)`;
        • detaches each copy by clearing `previous_controlnet`;
        • presents one sampler-facing ControlNet-like wrapper;
        • evaluates each detached ControlNet independently;
        • combines outputs by weighted additive streaming accumulation.

  - Composition is defined as:
        combined = Σ (w_i · α^i) · C_i(x)

    where:
        • C_i(x) is the independent output of ControlNet i;
        • w_i is the user-defined weight for that extracted chain position;
        • α is an order-bias term applied across the chain.

  - The single-ControlNet case intentionally remains native.  If the extracted
    or visible chain contains only one active ControlNet, there is no recursive
    multi-ControlNet traversal to remove, and the native path preserves the
    speed and behavior found in testing.

  - Because Composition receives completed conditioning rather than owning
    hint-image sockets, it does not raise missing-image errors.  It can,
    however, warn when visible non-zero weight rows do not correspond to an
    extracted upstream ControlNet, or when an upstream chain is longer than the
    visible `slot_count` and will be clipped.

  - This node is most useful after conventional Apply nodes have built a
    native ControlNet chain, especially in Flux or LoRA-heavy workflows where
    recursive multi-ControlNet execution can create excessive overhead or
    memory pressure.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in the core **ComfyUI**
    project:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

from .jlc_controlnet_nonrecursive_core import (
        safe_cnet_name,
)


MANIFEST = {
    "name": "JLC ControlNet Composition",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Converts an existing native ControlNet chain into a non-recursive "
        "weighted composition. Extracts and shallow-detaches upstream ControlNets, "
        "uses a sampler-facing composed wrapper for multi-ControlNet fusion, "
        "preserves native behavior for a single active ControlNet, and warns when "
        "visible weight rows do not match the extracted chain."
    ),
}


MAX_SLOTS = 10
DEBUG = True


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
        nonzero_visible_weight_rows = [i + 1 for i, w in enumerate(weights) if w != 0]
        warned = set()

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
                            "If this was not intentional, check upstream Apply/Orchestrator nodes, "
                            "disabled toggles, strength=0, or missing hint image connections."
                        ),
                    )
                return

            if chain_length < slot_count:
                unused_nonzero = [
                    row for row in range(chain_length + 1, slot_count + 1)
                    if weights[row - 1] != 0
                ]
                if unused_nonzero:
                    warn_once(
                        ("short_chain", chain_length, tuple(unused_nonzero)),
                        (
                            f"slot_count={slot_count} but extracted chain has only "
                            f"{chain_length} ControlNet(s). Unused non-zero weight rows: "
                            f"{unused_nonzero}. Extracted chain={chain_names}. If this was "
                            "not intentional, check upstream Apply/Orchestrator nodes, disabled "
                            "toggles, strength=0, or missing hint image connections."
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

        def process_conditioning(conditioning, stream_name):
            out = []

            for row_index, t in enumerate(conditioning):
                d = t[1].copy()
                cnet = d.get("control", None)

                if cnet is None:
                    warn_for_chain_length(0, stream_name, row_index, [])
                    if DEBUG:
                        print(
                            f"[JLC-ControlNet Composition] {stream_name}[{row_index}] "
                            "fallback=no_control"
                        )
                    out.append([t[0], d])
                    continue

                chain = extract_controlnet_chain(cnet)
                chain_names = [safe_cnet_name(c) for c in chain]
                warn_for_chain_length(len(chain), stream_name, row_index, chain_names)

                # Single-ControlNet path: leave native chain behavior untouched.
                if len(chain) <= 1:
                    if DEBUG:
                        print(
                            f"[JLC-ControlNet Composition] {stream_name}[{row_index}] "
                            f"fallback=native_single chain_order={chain_names}"
                        )
                    out.append([t[0], d])
                    continue

                detached_chain = make_detached_chain(chain)
                trimmed_chain = detached_chain[:slot_count]
                trimmed_names = [safe_cnet_name(c) for c in trimmed_chain]

                if len(trimmed_chain) <= 1:
                    if DEBUG:
                        print(
                            f"[JLC-ControlNet Composition] {stream_name}[{row_index}] "
                            f"fallback=trimmed_native_single original_chain={chain_names} "
                            f"trimmed_chain={trimmed_names}"
                        )
                    out.append([t[0], d])
                    continue

                trimmed_weights = weights[:len(trimmed_chain)]
                final_weights = [w * (alpha ** i) for i, w in enumerate(trimmed_weights)]

                if DEBUG:
                    print(
                        f"[JLC-ControlNet Composition] {stream_name}[{row_index}] "
                        f"fallback=composed original_chain={chain_names} "
                        f"trimmed_chain={trimmed_names} raw_weights={trimmed_weights} "
                        f"final_weights={final_weights} alpha={alpha}"
                    )

                d["control"] = JLC_ComposedControlNet(
                    trimmed_chain,
                    final_weights,
                    debug_label="JLC-ControlNet Composition",
                    debug_names=trimmed_names,
                    debug_alpha=alpha,
                )
                out.append([t[0], d])

            return out

        return (
            process_conditioning(positive, "positive"),
            process_conditioning(negative, "negative"),
        )
