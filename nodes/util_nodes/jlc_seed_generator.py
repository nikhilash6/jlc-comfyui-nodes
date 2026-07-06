"""
JLC Seed Generator
------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI** collection
    developed by **J. L. Córdova**.

- Purpose
    A shared seed source for workflows where multiple samplers or multiple
    inference stages should use the same seed for one full prompt execution.

    Typical use case:
        • one seed node feeds two or more KSampler seed inputs
        • stage 1 creates a latent
        • a stage-boundary cleanup node frees selected heavy model objects
        • stage 2 partially denoises the same latent using the same seed

- Reverse Seed Display Semantics
    This node intentionally presents queued seeds in the opposite user-facing
    sequence from normal ComfyUI seed widgets.

    Normal seed widgets usually mutate the visible seed after queue submission:
        seed = 1, control after generate = increment, queue count = 4
        visible widget advances toward 5 as queued prompts are prepared

    JLC Seed Generator instead preserves the user's visible base seed while the
    companion JavaScript display reports the seed actually used by each executed
    prompt:
        seed = 1, control after generate = increment, queue count = 4
        queued prompt 1 uses seed 1
        queued prompt 2 uses seed 2
        queued prompt 3 uses seed 3
        queued prompt 4 uses seed 4
        visible seed widget is restored to 1
        display shows the last seed used as each prompt executes

    The intent is to make parameter trials easier when seed consistency matters.
    A user can launch several queued variations, abort or adjust parameters, and
    still see the original base seed in the input widget without needing to
    remember or manually restore it.

- Display Spacer
    The final STRING widget is intentionally a harmless display-reservation row.
    The frontend companion turns it into a non-editing visual panel that shows
    the last seed used. If the frontend script is unavailable, it simply appears
    as an inert separator-like text field and is ignored by the backend.

- Design Notes
    This node uses ComfyUI's native seed widget control-after-generate behavior
    so the queued prompt data receives the correct per-run seed values. The
    frontend companion then restores the visible widget to the pre-queue base
    seed and displays the backend-reported seed actually used.

    The node returns both a small SEED-style dictionary and a plain INT. The
    INT output is convenient for ComfyUI seed inputs that expect integer seeds.

- Attribution & License
  - Concept and implementation by **J. L. Córdova** with development
    assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations
from typing import Any

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION

MAX_SEED = 0x7FFFFFFFFFFFFFFF  # 2^63 - 1

MANIFEST = {
    "name": "JLC Seed Generator",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Shared seed source for multi-sampler and multi-stage ComfyUI workflows. "
        "Uses ComfyUI's native seed queue behavior while the frontend restores "
        "the visible base seed and displays the last seed actually used."
    ),
}


class JLC_SeedGenerator:
    """Shared seed source with frontend-restored stable base seed display."""

    FUNCTION = "generator"
    CATEGORY = "utils/seed"

    RETURN_TYPES = ("SEED", "INT")
    RETURN_NAMES = ("seed", "seed_int")
    OUTPUT_NODE = False

    def __init__(self):
        self.last_seed: int | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": (
                            "Stable base seed input. During queued runs, ComfyUI may use "
                            "incremented, decremented, or randomized seeds, but this visible "
                            "input is restored to the value you entered. The display row below "
                            "shows the last seed actually used."
                        ),
                    },
                ),
                # Reserved display row for the frontend seed panel. Keep this
                # as a real widget, not hidden, so LiteGraph accounts for its
                # vertical space and does not easily clip the custom display.
                "spacer": (
                    "STRING",
                    {
                        "default": "──────── seed display ────────",
                        "multiline": False,
                        "tooltip": (
                            "This row shows the last seed actually used; the seed input above "
                            "remains the starting seed."
                        ),
                    },
                ),
            }
        }

    @staticmethod
    def _coerce_seed(seed: Any) -> int:
        if seed is None:
            seed = 0

        try:
            value = int(seed)
        except Exception:
            value = 0

        return max(0, min(value, MAX_SEED))

    def generator(self, seed=0, spacer=None):
        current_seed = self._coerce_seed(seed)
        self.last_seed = current_seed

        seed_text = str(current_seed)

        return {
            "result": (
                {"seed": current_seed},
                current_seed,
            ),
            # ComfyUI forwards these values to node.onExecuted(message).
            # Use list values for compatibility with common frontend examples.
            "ui": {
                "jlc_seed": [seed_text],
                "seed": [seed_text],
            },
        }


NODE_CLASS_MAPPINGS = {
    "JLC_SeedGenerator": JLC_SeedGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_SeedGenerator": "JLC Seed Generator",
}
