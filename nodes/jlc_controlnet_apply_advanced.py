"""
JLC ControlNet Apply (Advanced)

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
  - The **JLC ControlNet Apply (Advanced)** node applies a ControlNet
    to both positive and negative conditioning streams.

  - This version integrates **ControlNet loading directly into the node**
    while preserving ComfyUI's stateless execution model.

  - The node supports two ControlNet sources:
        • Upstream `control_net` input (preferred for chaining)
        • Internal loading via `control_net_name` dropdown

  - ControlNet source priority:
        1. If `control_net` input is connected → reuse the provided object
        2. Otherwise → load from `control_net_name`

  - This enables efficient **daisy-chained ControlNet workflows**, where:
        • The model is loaded once at the start of the chain
        • Subsequent nodes reuse the same ControlNet via pass-through
        • No global caching or shared state is introduced

  - When disabled (or strength = 0):
        • No ControlNet is loaded
        • All inputs pass through unchanged

  - This design avoids unnecessary model loads within a workflow
    while remaining fully deterministic and aligned with ComfyUI's
    memory and execution model.

    
- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Adapted from the **ControlNetApply** node in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

MANIFEST = {
    "name": "JLC ControlNet Apply (Advanced)",
    "version": (1, 1, 2),
    "author": "J. L. Córdova",
    "description": (
        "ControlNet apply node with integrated loader and lazy loading "
        "to eliminate unnecessary memory usage in chained workflows."
    ),
}

import folder_paths
import comfy.controlnet

# Optional debug flag
DEBUG = True

class JLC_ControlNetApplyAdvanced:
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable or disable this ControlNet. Disabled = no model load."
                }),

                "image": ("IMAGE", {
                    "tooltip": "Control image used to generate ControlNet conditioning."
                }),

                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),

                "vae": ("VAE",),

                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "ControlNet influence strength. 0 = disabled behavior."
                }),

                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "When ControlNet starts influencing diffusion."
                }),

                "end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "When ControlNet stops influencing diffusion."
                }),
            },

            "optional": {
                "control_net": ("CONTROL_NET", {
                    "tooltip": "Optional upstream ControlNet. Overrides dropdown selection."
                }),

                "control_net_name": (
                    folder_paths.get_filename_list("controlnet"),
                    {
                        "tooltip": (
                            "Select ControlNet model.\n"
                            "If the ControlNet input connector is used, this dropdown is ignored."
                        )
                    }
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "VAE", "CONTROL_NET")
    RETURN_NAMES = ("positive", "negative", "vae", "control_net")

    def apply_controlnet(
        self,
        enabled,
        image,
        positive,
        negative,
        vae,
        strength,
        start_percent,
        end_percent,
        control_net=None,
        control_net_name=None,
        extra_concat=None,
    ):
        # 🚫 HARD EXIT → no load, no memory usage
        if (not enabled) or strength == 0:
            return (positive, negative, vae, control_net)

        if extra_concat is None:
            extra_concat = []

            # 🔁 Resolve ControlNet source

            if control_net is not None:
                if DEBUG:
                    print("[JLC-ControlNet] 🔁 Reusing ControlNet via input connection")

            else:
                # 🔒 Validate dropdown selection (handles None / "" edge cases)
                if not control_net_name:
                    raise RuntimeError("No ControlNet provided or selected.")

                if DEBUG:
                    print(f"[JLC-ControlNet] ✅ Loading ControlNet '{control_net_name}'")

                controlnet_path = folder_paths.get_full_path_or_raise(
                    "controlnet",
                    control_net_name
                )

                control_net = comfy.controlnet.load_controlnet(controlnet_path)

                if control_net is None:
                    raise RuntimeError("❌ No ControlNet provided or selected.")
                
        # ---- Core logic (unchanged from original node) ----

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in (positive, negative):
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(
                        control_hint,
                        strength,
                        (start_percent, end_percent),
                        vae=vae,
                        extra_concat=extra_concat
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)

        return (out[0], out[1], vae, control_net)