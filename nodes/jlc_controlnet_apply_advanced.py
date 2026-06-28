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

    - This version integrates ControlNet loading and reuse through the
      shared JLC model residency cache core instead of maintaining a
      node-local cache.

    - The node supports two ControlNet sourcing modes:
            • Upstream `control_net` input (direct chained reuse)
            • Internal loading via `control_net_name` dropdown, with
              process-local reuse handled by `jlc_model_cache_core`

    - ControlNet source priority:
            1. If `control_net` input is connected → reuse the provided object
            2. Otherwise → load/reuse the selected `control_net_name` through
               the shared JLC model cache core

    - When disabled (or strength = 0):
            • No ControlNet is loaded
            • All inputs pass through unchanged

    - This design:
            • Preserves ComfyUI's stateless graph execution model
            • Keeps ControlNet application logic aligned with ComfyUI's native
              Apply ControlNet behavior
            • Reuses resident ControlNet base models through a shared,
              family-aware LRU cache
            • Leaves sampler-facing ControlNet copies isolated per conditioning
              chain exactly as before

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Adapted from the **ControlNetApply** node in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

MANIFEST = {
    "name": "JLC ControlNet Apply (Advanced)",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "ControlNet apply node with integrated loader using the shared JLC "
        "model residency cache core."
    ),
}

import os

import folder_paths
import comfy.controlnet

try:
    from .engines.jlc_model_cache_core import (
        cache_info,
        get_controlnet_cache_capacity,
        get_or_load_model,
        make_controlnet_cache_key,
    )
except ImportError:
    from .engines.jlc_model_cache_core import (  # type: ignore
        cache_info,
        get_controlnet_cache_capacity,
        get_or_load_model,
        make_controlnet_cache_key,
    )


# Optional debug flag. Leave True for maintenance/testing; set False if console
# output becomes too noisy for release builds.
DEBUG = True

NONE_CONTROLNET_LABEL = "NONE / Input Override"
CONTROLNET_CACHE_FAMILY = "controlnet"
CONTROLNET_CACHE_ROLE = "model"
CONTROLNET_CACHE_POLICY = "lru_family_capacity"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _get_cnet_name(path=None, name=None):
    if path:
        return os.path.splitext(os.path.basename(str(path)))[0]
    if name:
        return os.path.splitext(os.path.basename(str(name)))[0]
    return "external"


def _cleanup_controlnet_for_cache(control_net):
    """Best-effort unload callback used by the shared cache on eviction."""

    cleanup = getattr(control_net, "cleanup", None)
    if callable(cleanup):
        cleanup()


def _load_controlnet_from_disk(controlnet_path, cnet_display):
    if DEBUG:
        print(f"[JLC-ControlNet] 💾 Loading ControlNet '{cnet_display}' from disk")

    control_net = comfy.controlnet.load_controlnet(controlnet_path)

    if control_net is None:
        raise RuntimeError(f"❌ Invalid ControlNet model file: {controlnet_path}")

    # Ensure the reuse safeguard has a predictable initial state.
    try:
        control_net._jlc_dirty = False
    except Exception:
        pass

    return control_net


def _resolve_controlnet_from_cache(control_net_name):
    controlnet_path = folder_paths.get_full_path_or_raise(
        "controlnet",
        control_net_name,
    )
    cnet_display = _get_cnet_name(path=controlnet_path)
    cache_key = make_controlnet_cache_key(controlnet_path)

    control_net = get_or_load_model(
        cache_key,
        lambda: _load_controlnet_from_disk(controlnet_path, cnet_display),
        family=CONTROLNET_CACHE_FAMILY,
        model_path=controlnet_path,
        role=CONTROLNET_CACHE_ROLE,
        unload_fn=_cleanup_controlnet_for_cache,
        policy=CONTROLNET_CACHE_POLICY,
    )

    # SELF-HEALING SAFEGUARD:
    # The cached resident object is cleaned before reuse if a prior run marked
    # it dirty. Per-conditioning/sampler-facing objects are still produced with
    # `control_net.copy()` below, just like ComfyUI's native Apply ControlNet.
    if getattr(control_net, "_jlc_dirty", False):
        if DEBUG:
            print(f"[JLC-ControlNet] 🧽 Cleaning cached ControlNet '{cnet_display}'")
        cleanup = getattr(control_net, "cleanup", None)
        if callable(cleanup):
            cleanup()
        control_net._jlc_dirty = False

    if DEBUG:
        capacity = get_controlnet_cache_capacity()
        info = cache_info()
        names = [
            _get_cnet_name(path=entry.get("model_path"))
            for entry in info.get("entries", [])
            if entry.get("family") == CONTROLNET_CACHE_FAMILY
        ]
        print(f"[JLC-ControlNet] 🧠 Shared cache ({len(names)}/{capacity}): {names}")

    return control_net, cnet_display


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
                    [NONE_CONTROLNET_LABEL] + folder_paths.get_filename_list("controlnet"),
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
        # HARD EXIT: do not resolve or load any ControlNet while idle.
        if (not enabled) or strength == 0:
            if DEBUG:
                print(f"[JLC-ControlNet] 😴 Node idle (enabled={enabled})")
            return (positive, negative, vae, control_net)

        if control_net_name == NONE_CONTROLNET_LABEL:
            control_net_name = None

        if control_net is None and control_net_name is None:
            if DEBUG:
                print("[JLC-ControlNet] ⏭️ Skipped (no input, no model selected)")
            return (positive, negative, vae, control_net)

        if DEBUG:
            print(f"[JLC-ControlNet] 🟢 Node triggered (enabled={enabled}, strength={strength})")

        if extra_concat is None:
            extra_concat = []

        # ------------------------------------------------------------------
        # Resolve ControlNet source
        # ------------------------------------------------------------------

        if control_net is not None:
            cnet_display = _get_cnet_name(name=control_net_name)
            if DEBUG:
                print(f"[JLC-ControlNet] 🔌 Using ControlNet '{cnet_display}' from input")
        else:
            control_net, cnet_display = _resolve_controlnet_from_cache(control_net_name)

        # ------------------------------------------------------------------
        # Core logic based on ComfyUI's native Apply ControlNet node
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

        # Mark the resident/base object dirty after use so the next cache reuse
        # can call cleanup() before creating fresh sampler-facing copies.
        try:
            control_net._jlc_dirty = True
        except Exception:
            pass

        return (out[0], out[1], vae, control_net)
