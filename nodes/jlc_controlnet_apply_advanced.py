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

  - This version introduces **integrated ControlNet loading** and
    **lazy evaluation behavior**, allowing the node to:
        • Load a ControlNet internally via dropdown selection
        • Accept a ControlNet from upstream nodes
        • Avoid loading any model when disabled

  - This enables efficient **daisy-chained ControlNet pipelines**
    without unnecessary memory usage.

  - ControlNet source priority:
        1. If `control_net` input is connected → use it
        2. Otherwise → load from `control_net_name`

  - When disabled (or strength = 0):
        • No ControlNet is loaded
        • All inputs pass through unchanged

  - This solves a key limitation in ComfyUI where loader nodes
    allocate memory even when downstream logic is bypassed.


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
    "version": (1, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "ControlNet apply node with integrated loader and lazy loading "
        "to eliminate unnecessary memory usage in chained workflows."
    ),
}

import folder_paths
import comfy.controlnet

# Global Cache to avoid repeated ControlNet models loads
GLOBAL_CONTROLNET_CACHE = {}

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
    ):
        # 🚫 HARD EXIT → no load, no memory usage
        if (not enabled) or strength == 0:
            return (positive, negative, vae, control_net)

        # 🔁 Resolve ControlNet source
        if control_net is None:
            if control_net_name is None:
                raise RuntimeError("No ControlNet provided or selected.")

            controlnet_path = folder_paths.get_full_path_or_raise(
                "controlnet",
                control_net_name
            )

            # ⚡ Cache lookup
            if controlnet_path in GLOBAL_CONTROLNET_CACHE:
                control_net = GLOBAL_CONTROLNET_CACHE[controlnet_path]
            else:
                control_net = comfy.controlnet.load_controlnet(controlnet_path)
                if control_net is None:
                    raise RuntimeError("Invalid ControlNet model file.")
                GLOBAL_CONTROLNET_CACHE[controlnet_path] = control_net

        # ---- Core logic (unchanged from your original node) ----

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in (positive, negative):
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get("control", None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = (
                        control_net.copy()
                        .set_cond_hint(
                            control_hint,
                            strength,
                            (start_percent, end_percent),
                            vae,
                        )
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                c.append([t[0], d])

            out.append(c)

        return (out[0], out[1], vae, control_net)


NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetApplyAdvanced": JLC_ControlNetApplyAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetApplyAdvanced": "JLC ControlNet Apply (Advanced)",
}