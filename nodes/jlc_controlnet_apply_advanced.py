"""
JLC ControlNet Apply (Advanced)
-------------------------------

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
  - The **JLC ControlNet Apply (Advanced)** node applies one ControlNet to
    both positive and negative conditioning streams, with optional internal
    ControlNet loading through the shared JLC model residency cache.

  - The node supports two ControlNet sourcing modes:
        • upstream `control_net` input for explicit wired reuse;
        • internal dropdown loading through `control_net_name`.

  - Source priority is deterministic:
        1. if the upstream `control_net` input is connected, it is used;
        2. otherwise the selected dropdown model is loaded or reused through
           the shared JLC cache.

  - The shared cache owns only raw resident base ControlNet objects.  Per-run
    conditioning state is always applied to isolated `.copy()` instances, and
    cached base objects are never passed through `set_cond_hint()`.

  - Native ComfyUI chain semantics are preserved:
        • the hint image is a required input; an active node explicitly rejects
          a runtime `None` value with a clear error before model loading or tensor work;
        • previous ControlNets already attached to conditioning are preserved
          through `set_previous_controlnet(prev_cnet)`;
        • disabled or zero-strength operation exits before any dropdown model
          is loaded;
        • the node does not perform non-recursive fusion itself.

  - In the larger JLC non-recursive ControlNet workflow, this node can build
    ordinary ControlNet chains efficiently, while downstream Composition or
    Orchestrator nodes may perform explicit non-recursive weighted fusion.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Adapted from the **ControlNetApplyAdvanced** node in the core
    **ComfyUI** project:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""


import os

import folder_paths
import comfy.controlnet

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

from .engines.jlc_model_cache_core import (
        get_controlnet_cache_capacity,
        get_or_load_model,
        make_controlnet_cache_key,
)

MANIFEST = {
    "name": "JLC ControlNet Apply (Advanced)",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "ControlNet apply node with optional internal dropdown loading through "
        "the shared JLC model residency cache. Preserves native ControlNet chain "
        "semantics, applies conditioning only to per-run copies, rejects a null "
        "runtime hint on active execution, and avoids model loading when disabled "
        "or strength is zero."
    ),
}

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

        if image is None:
            message = (
                "JLC ControlNet Apply (Advanced): active execution received None for "
                "the required image input. This usually means the image socket is "
                "disconnected or is linked to a DISABLED/hidden JLC Aux Wrapper "
                "output. Connect a valid hint image, or disable this Apply node / "
                "set strength to 0."
            )
            print(f"[JLC-ControlNet Apply Advanced][ERROR] {message}")
            raise RuntimeError(message)

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

        capacity = max(0, int(get_controlnet_cache_capacity()))

        control_net = get_or_load_model(
            cache_key,
            loader,
            family=CONTROLNET_CACHE_FAMILY,
            model_path=controlnet_path,
            role=CONTROLNET_CACHE_ROLE,
            policy=CONTROLNET_CACHE_POLICY,
            max_loaded_for_family=capacity,
        )

        if DEBUG:
            print(
                f"[JLC-ControlNet] Shared cache family='{CONTROLNET_CACHE_FAMILY}' "
                f"capacity={capacity}"
            )

        return control_net


def _display_name(path=None, name=None):
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    if name:
        return os.path.splitext(os.path.basename(name))[0]
    return "external"
