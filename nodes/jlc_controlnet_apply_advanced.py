"""
JLC ControlNet Apply (Advanced)

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

- Node Purpose
    - Applies a ControlNet to both positive and negative conditioning streams.
    - Supports optional upstream CONTROL_NET input or internal dropdown loading.
    - Uses the shared JLC model residency cache for internally loaded ControlNet
      base objects only.

- Safety / Architecture Notes
    - The shared cache owns only resident base ControlNet objects.
    - Conditioning is always applied to per-execution `.copy()` instances.
    - The cached base object is never passed through `set_cond_hint()`.
    - No `_jlc_dirty` marker is needed because cached residents remain raw bases.
    - Native Comfy chain semantics are preserved via `set_previous_controlnet()`.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Adapted from the **ControlNetApplyAdvanced** node in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

MANIFEST = {
    "name": "JLC ControlNet Apply (Advanced)",
    "version": (1, 1, 3),
    "author": "J. L. Córdova",
    "description": (
        "ControlNet apply node with integrated dropdown loading through the "
        "shared JLC model residency cache. Preserves native ControlNet chain "
        "semantics and keeps cached objects as raw base residents."
    ),
}

import os

import folder_paths
import comfy.controlnet

try:
    from .engines.jlc_model_cache_core import (
        make_controlnet_cache_key,
        get_or_load_model,
        cache_keys,
        get_family_capacity,
    )
except ImportError:
    from jlc_model_cache_core import (  # type: ignore
        make_controlnet_cache_key,
        get_or_load_model,
        cache_keys,
        get_family_capacity,
    )


DEBUG = True
CONTROLNET_CACHE_FAMILY = "controlnet"
CONTROLNET_CACHE_ROLE = "controlnet_base"
CONTROLNET_CACHE_POLICY = "lru_family_capacity"
CONTROLNET_NONE_LABEL = "NONE / Input Override"


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
                    [CONTROLNET_NONE_LABEL] + folder_paths.get_filename_list("controlnet"),
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
        # Hard bypass before any dropdown resolution or model loading.
        if (not enabled) or strength == 0:
            if DEBUG:
                print(f"[JLC-ControlNet] Node idle (enabled={enabled}, strength={strength})")
            return (positive, negative, vae, control_net)

        if control_net_name == CONTROLNET_NONE_LABEL:
            control_net_name = None

        if control_net is None and control_net_name is None:
            if DEBUG:
                print("[JLC-ControlNet] Skipped (no input ControlNet and no dropdown model selected)")
            return (positive, negative, vae, control_net)

        if extra_concat is None:
            extra_concat = []

        # Resolve ControlNet source. Upstream input always wins and is never
        # registered into the shared cache.
        if control_net is not None:
            if DEBUG:
                print("[JLC-ControlNet] Using ControlNet from input")
        else:
            control_net = self._load_controlnet_from_shared_cache(control_net_name)

        # ------------------------------------------------------------------
        # Core logic based on ComfyUI's native ControlNetApplyAdvanced node.
        # This builds native previous_controlnet chains; it does not compose.
        # ------------------------------------------------------------------
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
                    # The cached/control input object is treated as a raw base.
                    # Conditioning state is applied only to this isolated copy.
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

        return (out[0], out[1], vae, control_net)

    def _load_controlnet_from_shared_cache(self, control_net_name):
        if control_net_name is None:
            raise RuntimeError("No ControlNet provided or selected.")

        controlnet_path = folder_paths.get_full_path_or_raise(
            "controlnet",
            control_net_name,
        )
        cache_key = make_controlnet_cache_key(controlnet_path)
        display_name = _display_name(path=controlnet_path)

        def loader():
            print(f"[JLC-ControlNet] Loading ControlNet '{display_name}' from disk")
            cnet = comfy.controlnet.load_controlnet(controlnet_path)
            if cnet is None:
                raise RuntimeError(f"Invalid ControlNet model file: {control_net_name}")

            # Defensive state hygiene: a newly loaded cached base should not
            # inherit accidental MultiGPU clones. Real MultiGPU support is not
            # part of this node's cache/conditioning responsibility.
            if hasattr(cnet, "multigpu_clones"):
                cnet.multigpu_clones = {}
            return cnet

        control_net = get_or_load_model(
            cache_key,
            loader,
            family=CONTROLNET_CACHE_FAMILY,
            model_path=controlnet_path,
            role=CONTROLNET_CACHE_ROLE,
            policy=CONTROLNET_CACHE_POLICY,
        )

        if DEBUG:
            capacity = get_family_capacity(CONTROLNET_CACHE_FAMILY)
            resident = cache_keys(family=CONTROLNET_CACHE_FAMILY)
            print(
                f"[JLC-ControlNet] Shared cache family='{CONTROLNET_CACHE_FAMILY}' "
                f"resident={len(resident)} capacity={capacity}"
            )

        return control_net


def _display_name(path=None, name=None):
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    if name:
        return os.path.splitext(os.path.basename(name))[0]
    return "external"
