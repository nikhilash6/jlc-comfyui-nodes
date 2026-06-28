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

- Release Notes (v1.2.2)
    - Uses the shared JLC ControlNet composition core so this node and the
      Advanced Orchestrator cannot drift apart.
    - Adds a conservative dynamic weight-count widget for frontend visibility.
    - Predeclares ten weight widgets but defaults to the prior five-weight
      behavior for backwards compatibility.
    - `slot_count` is authoritative for how many chain entries / weights are
      considered by the backend; hidden higher weights remain serialized but
      are ignored.
    - Core non-recursive fusion math is unchanged.

- Performance Notes
    - Reduces recursive ControlNet traversal overhead.
    - Typically improves runtime and stability in multi-ControlNet workflows.
    - Performance gains scale with number of ControlNets (N ≥ 2).

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the MIT License.
"""

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

MANIFEST = {
    "name": "JLC ControlNet Composition",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Non-recursive ControlNet composition using explicit weighted fusion. "
        "Provides deterministic, order-invariant behavior when alpha=1, with "
        "correct native passthrough for single-ControlNet cases. Uses the "
        "shared JLC ControlNet composition core so this node and the Advanced "
        "Orchestrator cannot drift apart. Adds dynamic visibility for up to ten "
        "weight slots while preserving the prior five-weight default. Fusion "
        "math is unchanged."
    ),
}

try:
    from .jlc_controlnet_composition_core import (
        JLC_ComposedControlNet,
        extract_controlnet_chain,
        make_detached_chain,
    )
except ImportError:  # Supports direct loose-module imports during local testing.
    from jlc_controlnet_composition_core import (  # type: ignore
        JLC_ComposedControlNet,
        extract_controlnet_chain,
        make_detached_chain,
    )


MAX_COMPOSITION_WEIGHTS = 10
DEFAULT_VISIBLE_WEIGHTS = 5


# ------------------------------------------------------------
# 🧠 Shared composition core
# ------------------------------------------------------------
# JLC_ComposedControlNet, extract_controlnet_chain, and make_detached_chain are
# imported from jlc_controlnet_composition_core.py. The non-recursive fusion
# algorithm is intentionally centralized there without changing its math.


def normalize_slot_count(slot_count, *, default=DEFAULT_VISIBLE_WEIGHTS, max_slots=MAX_COMPOSITION_WEIGHTS):
    """Clamp dynamic weight-slot count while preserving old five-weight default."""
    try:
        count = int(slot_count)
    except (TypeError, ValueError):
        count = int(default)

    return max(1, min(int(max_slots), count))


def slot_count_widget():
    return (
        "INT",
        {
            "default": DEFAULT_VISIBLE_WEIGHTS,
            "min": 1,
            "max": MAX_COMPOSITION_WEIGHTS,
            "step": 1,
            "display": "number",
            "tooltip": (
                "Authoritative number of ControlNet weights to expose and use. "
                "Weights above this value remain serialized but are ignored. "
                "Default 5 preserves the original Composition node behavior."
            ),
        },
    )


def weight_widget(index):
    return (
        "FLOAT",
        {
            "default": 1.0,
            "min": -10.0,
            "max": 10.0,
            "step": 0.01,
            "tooltip": f"Contribution of ControlNet {index} (can be negative)",
        },
    )


def build_weight_widgets(max_slots=MAX_COMPOSITION_WEIGHTS):
    return {f"weight_{i}": weight_widget(i) for i in range(1, int(max_slots) + 1)}


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
                "slot_count": slot_count_widget(),
                **build_weight_widgets(MAX_COMPOSITION_WEIGHTS),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Order bias. <1 favors earlier ControlNets, >1 favors later ones",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")

    def compose_controlnet(
        self,
        positive,
        negative,
        slot_count=DEFAULT_VISIBLE_WEIGHTS,
        weight_1=1.0,
        weight_2=1.0,
        weight_3=1.0,
        weight_4=1.0,
        weight_5=1.0,
        weight_6=1.0,
        weight_7=1.0,
        weight_8=1.0,
        weight_9=1.0,
        weight_10=1.0,
        alpha=1.0,
    ):
        count = normalize_slot_count(slot_count)
        all_weights = [
            weight_1,
            weight_2,
            weight_3,
            weight_4,
            weight_5,
            weight_6,
            weight_7,
            weight_8,
            weight_9,
            weight_10,
        ]
        weights = all_weights[:count]

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

                # 🧹 Early filtering (matches JLC ControlNet Orchestrator discipline)
                prepared = []
                prepared_weights = []

                for c, w in zip(trimmed_chain, weights[:len(trimmed_chain)]):
                    if c is None:
                        continue
                    if w == 0:
                        continue
                    prepared.append(c)
                    prepared_weights.append(w)

                # 🔸 Pure passthrough (Edge and aberrant cases, like all weights = 0)
                if not prepared:
                    out.append([t[0], d])
                    continue

                # 🟢 Single-ControlNet fallback
                if len(prepared) == 1:
                    d["control"] = prepared[0]
                    out.append([t[0], d])
                    continue

                # 🎯 Apply alpha AFTER filtering
                final_weights = [
                    w * (alpha ** i)
                    for i, w in enumerate(prepared_weights)
                ]

                print(f"[JLC-ControlNet] ⚙️ weights={final_weights} alpha={alpha}")

                # 🧩 Compose only when N ≥ 2
                d["control"] = JLC_ComposedControlNet(
                    prepared,
                    final_weights,
                )

                out.append([t[0], d])

            return out

        new_positive = process_conditioning(positive)
        new_negative = process_conditioning(negative)

        return (new_positive, new_negative)


NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetComposition": JLC_ControlNetComposition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetComposition": "⚙️ JLC ControlNet Composition",
}
