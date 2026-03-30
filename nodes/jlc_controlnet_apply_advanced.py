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

    - This version integrates **ControlNet loading, reuse, and session-level caching**
        directly into the node while preserving ComfyUI's stateless execution model.

    - The node supports three ControlNet sourcing modes:
            • Upstream `control_net` input (direct chained reuse)
            • Internal LRU cache (wireless reuse across nodes/runs)
            • Internal loading via `control_net_name` dropdown

    - ControlNet source priority:
            1. If `control_net` input is connected → reuse the provided object
            2. Otherwise, if present in cache → reuse cached object (wireless)
            3. Otherwise → load from disk via `control_net_name`

    - The node implements a **bounded LRU cache**:
            • Stores a limited number of ControlNet models in system RAM
            • Automatically evicts least recently used models
            • Prevents unbounded memory growth

    - Cached ControlNet objects are protected by a **self-healing safeguard**:
            • Models are marked as “dirty” after use
            • Cached instances are cleaned (`cleanup()`) before reuse
            • Prevents reuse of mutated ControlNet state

    - This enables efficient **daisy-chained and cross-node workflows**, where:
            • Models can be reused via explicit wiring (deterministic)
            • Models can also be reused wirelessly when still cached
            • Redundant disk loads are minimized without unsafe global state

    - When disabled (or strength = 0):
            • No ControlNet is loaded
            • All inputs pass through unchanged

    - This design:
            • Avoids unsafe global mutation
            • Maintains deterministic behavior within execution chains
            • Leverages ComfyUI's native VRAM management (no interference)
            • Improves performance through safe, bounded reuse

   
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

import os
import folder_paths
import comfy.controlnet

from collections import OrderedDict

# 🔒 Strong cache (session-level) with eviction policy
GLOBAL_CONTROLNET_CACHE = OrderedDict()
MAX_CACHED_CONTROLNETS = 3

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
                    ["NONE / Input Override"] + folder_paths.get_filename_list("controlnet"),
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
        # 🚫 HARD EXIT
        if (not enabled) or strength == 0:
            print(f"[JLC-ControlNet] 😴 Node idle (enabled={enabled})")
            return (positive, negative, vae, control_net)
        
        if control_net_name == "NONE / Input Override":
            control_net_name = None
        
        if control_net is None and control_net_name is None:
            if DEBUG:
                print("[JLC-ControlNet] ⏭️ Skipped (no input, no model selected)")
            return (positive, negative, vae, control_net)

        print(f"[JLC-ControlNet] 🟢 Node triggered (enabled={enabled}, strength={strength})")

        if extra_concat is None:
            extra_concat = []

        def _get_cnet_name(path=None, name=None):
            if path:
                return os.path.splitext(os.path.basename(path))[0]
            if name:
                return os.path.splitext(os.path.basename(name))[0]
            return "external"

        # 🔁 Resolve ControlNet source
        if control_net is not None:
            cnet_display = _get_cnet_name(name=control_net_name)
            print(f"[JLC-ControlNet] 🔌 Using ControlNet '{cnet_display}' from input")

        else:
            if control_net_name is None:
                raise RuntimeError("❌ No ControlNet provided or selected.")

            controlnet_path = folder_paths.get_full_path_or_raise(
                "controlnet",
                control_net_name
            )

            cnet_display = _get_cnet_name(path=controlnet_path)

            # 🔒 Check strong cache
            if controlnet_path in GLOBAL_CONTROLNET_CACHE:
                control_net = GLOBAL_CONTROLNET_CACHE[controlnet_path]

                # 🔁 mark as recently used (LRU)
                GLOBAL_CONTROLNET_CACHE.move_to_end(controlnet_path)

                print(f"[JLC-ControlNet] 🔁 ControlNet '{cnet_display}' in selector already in cache")

                # 🧠 SELF-HEALING SAFEGUARD (only on reuse)
                if getattr(control_net, "_jlc_dirty", False):
                    print(f"[JLC-ControlNet] 🧽 Cleaning cached ControlNet '{cnet_display}'")
                    control_net.cleanup()
                    control_net._jlc_dirty = False

            else:
                print(f"[JLC-ControlNet] 💾 Loading ControlNet '{cnet_display}' from disk")

                control_net = comfy.controlnet.load_controlnet(controlnet_path)

                if control_net is None:
                    raise RuntimeError("❌ Invalid ControlNet model file.")

                GLOBAL_CONTROLNET_CACHE[controlnet_path] = control_net

                # 🔥 Eviction (LRU)
                if len(GLOBAL_CONTROLNET_CACHE) > MAX_CACHED_CONTROLNETS:
                    evicted_key, _ = GLOBAL_CONTROLNET_CACHE.popitem(last=False)
                    evicted_name = _get_cnet_name(path=evicted_key)
                    print(f"[JLC-ControlNet] 🗑️ Evicted '{evicted_name}' from cache")

        # 🔍 Cache debug
        cache_keys = list(GLOBAL_CONTROLNET_CACHE.keys())
        cache_names = [_get_cnet_name(path=k) for k in cache_keys]
        print(f"[JLC-ControlNet] 🧠 Cache ({len(cache_names)}/{MAX_CACHED_CONTROLNETS}): {cache_names}")
        

        # ------------------------------------------------------------------
        # ---- Core logic based on Comfy's native node Apply ControlNet ----

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
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
                            vae=vae,
                            extra_concat=extra_concat,
                        )
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                c.append([t[0], d])

            out.append(c)

        # 🧠 MARK AS DIRTY (post-use)
        control_net._jlc_dirty = True

        return (out[0], out[1], vae, control_net)