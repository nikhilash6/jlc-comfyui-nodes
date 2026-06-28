"""
JLC Stage Boundary VRAM Cleanup
-------------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI** collection
    developed by **J. L. Córdova**.

- Experimental Warning
    This is an experimental utility node.

    It is intended for advanced multi-stage workflows where the user
    deliberately wants to free selected heavy model objects after a stage
    boundary. It may affect model residency, reload behavior, execution time,
    and VRAM usage in ways that depend on ComfyUI's current internal model
    management behavior.

    Use only when the workflow is structured so that the upstream model objects
    are no longer needed after the LATENT passthrough point. ComfyUI remains the
    authority for model lifecycle management, and this node should be treated as
    a best-effort helper rather than a guaranteed VRAM reset.

- Purpose
    A latent-triggered stage-boundary cleanup node for multi-stage workflows.

    Typical use case:
        Stage 1: load/use a large base model, inpaint model, or JLC-managed
                 ControlNet stack to create a base latent.
        Boundary: this node receives the latent, evicts selected heavy objects,
                  clears allocator leftovers, and passes the latent onward.
        Stage 2: load/use another model family for partial denoising.

    This is intentionally not a generic CLIP/VAE cleanup node. The robust
    targets are:
        • a connected ComfyUI MODEL object and its clones/additional models
        • all currently loaded ComfyUI models, when explicitly requested
        • JLC-managed ControlNet resident cache entries
        • all JLC-managed resident cache entries, when explicitly requested
        • final best-effort allocator cleanup

- Design Notes
    ComfyUI remains the authority for its own model management. This node only
    calls ComfyUI's public-ish model_management helpers when available, and it
    uses the JLC shared model cache core for JLC-owned resident models.

    The node is a passthrough: it returns the same LATENT it receives.

- Attribution & License
  - Concept and implementation by **J. L. Córdova** with development
    assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION

MANIFEST = {
    "name": "JLC Stage Boundary VRAM Cleanup",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Latent-triggered stage-boundary cleanup node for multi-stage ComfyUI "
        "workflows. Can unload a connected MODEL through ComfyUI model "
        "management, evict JLC ControlNet cache entries, optionally unload all "
        "ComfyUI/JLC resident models, and clear CUDA allocator leftovers."
    ),
}

import gc
import time
from typing import Any, Optional

try:
    import comfy.model_management as model_management
except Exception as exc:  # pragma: no cover - depends on ComfyUI runtime
    model_management = None
    _MODEL_MANAGEMENT_IMPORT_ERROR = exc
else:
    _MODEL_MANAGEMENT_IMPORT_ERROR = None


try:
    from ..engines.jlc_model_cache_core import (
        cuda_cleanup as jlc_cuda_cleanup,
        evict_family as jlc_evict_family,
        unload_all as jlc_unload_all,
    )
except Exception as exc:  # pragma: no cover - import layout fallback
    _JLC_CACHE_IMPORT_ERROR = exc
    try:
        # Fallback for unusual import contexts during development/testing.
        from nodes.engines.jlc_model_cache_core import (  # type: ignore
            cuda_cleanup as jlc_cuda_cleanup,
            evict_family as jlc_evict_family,
            unload_all as jlc_unload_all,
        )
    except Exception:
        jlc_cuda_cleanup = None
        jlc_evict_family = None
        jlc_unload_all = None
else:
    _JLC_CACHE_IMPORT_ERROR = None

# End of Import Section
# ###############################################################


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(f"[JLC Stage Boundary VRAM Cleanup] {message}")


def _warn(message: str) -> None:
    print(f"[JLC Stage Boundary VRAM Cleanup] Warning: {message}")


def _unwrap_model_patcher(model: Any) -> Optional[Any]:
    """
    Return the most likely ComfyUI ModelPatcher object.

    Standard ComfyUI MODEL sockets usually pass a ModelPatcher directly. This
    wrapper also accepts a few dict-shaped variants used by some custom nodes.
    """

    if model is None:
        return None

    if hasattr(model, "clone_base_uuid"):
        return model

    if isinstance(model, dict):
        for key in ("model", "model_patcher", "patcher", "unet"):
            candidate = model.get(key)
            if candidate is not None and hasattr(candidate, "clone_base_uuid"):
                return candidate

    # Last-resort return. ComfyUI will reject it if it is not a valid patcher,
    # and the caller will catch/log the failure.
    return model


def _soft_empty_cache(*, verbose: bool = True) -> bool:
    """Call ComfyUI's backend-aware soft cache cleanup when available."""

    if model_management is None:
        if _MODEL_MANAGEMENT_IMPORT_ERROR is not None:
            _warn(f"could not import comfy.model_management: {_MODEL_MANAGEMENT_IMPORT_ERROR}")
        return False

    soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
    if not callable(soft_empty_cache):
        _warn("comfy.model_management.soft_empty_cache() is unavailable.")
        return False

    try:
        try:
            soft_empty_cache(True)
        except TypeError:
            soft_empty_cache()
        _log("ComfyUI soft_empty_cache executed.", verbose=verbose)
        return True
    except Exception as exc:
        _warn(f"ComfyUI soft_empty_cache failed: {exc}")
        return False


def _unload_connected_model(model: Any, *, all_devices: bool, verbose: bool = True) -> bool:
    """Unload one connected MODEL and its clones/additional models if possible."""

    if model is None:
        _log("No MODEL input connected; targeted ComfyUI model unload skipped.", verbose=verbose)
        return False

    if model_management is None:
        if _MODEL_MANAGEMENT_IMPORT_ERROR is not None:
            _warn(f"could not import comfy.model_management: {_MODEL_MANAGEMENT_IMPORT_ERROR}")
        return False

    unload_model_and_clones = getattr(model_management, "unload_model_and_clones", None)
    if not callable(unload_model_and_clones):
        _warn("comfy.model_management.unload_model_and_clones() is unavailable.")
        return False

    target = _unwrap_model_patcher(model)
    if target is None:
        _log("MODEL input resolved to None; targeted unload skipped.", verbose=verbose)
        return False

    try:
        unload_model_and_clones(
            target,
            unload_additional_models=True,
            all_devices=bool(all_devices),
        )
        _log(
            "Requested targeted unload of connected MODEL, clones, and "
            f"additional models; all_devices={bool(all_devices)}.",
            verbose=verbose,
        )
        return True
    except TypeError:
        # Older/forked ComfyUI fallback if the helper exists but has a narrower
        # signature. Keep this conservative rather than inventing our own clone
        # matching logic.
        try:
            unload_model_and_clones(target)
            _log("Requested targeted unload of connected MODEL using fallback signature.", verbose=verbose)
            return True
        except Exception as exc:
            _warn(f"targeted ComfyUI MODEL unload failed with fallback signature: {exc}")
            return False
    except Exception as exc:
        _warn(f"targeted ComfyUI MODEL unload failed: {exc}")
        return False


def _unload_all_comfy_models(*, verbose: bool = True) -> bool:
    """Explicit hammer: ask ComfyUI to unload all currently resident models."""

    if model_management is None:
        if _MODEL_MANAGEMENT_IMPORT_ERROR is not None:
            _warn(f"could not import comfy.model_management: {_MODEL_MANAGEMENT_IMPORT_ERROR}")
        return False

    unload_all_models = getattr(model_management, "unload_all_models", None)
    if not callable(unload_all_models):
        _warn("comfy.model_management.unload_all_models() is unavailable.")
        return False

    try:
        unload_all_models()
        _log("Requested ComfyUI unload_all_models().", verbose=verbose)
        return True
    except Exception as exc:
        _warn(f"ComfyUI unload_all_models failed: {exc}")
        return False


def _evict_jlc_controlnet_cache(*, safe: bool, verbose: bool = True) -> int:
    """Evict JLC-managed ControlNet cache entries."""

    if jlc_evict_family is None:
        if _JLC_CACHE_IMPORT_ERROR is not None:
            _warn(f"could not import JLC cache core: {_JLC_CACHE_IMPORT_ERROR}")
        else:
            _warn("JLC cache core evict_family() is unavailable.")
        return 0

    try:
        count = int(
            jlc_evict_family(
                "controlnet",
                reason="stage_boundary_controlnet_evict",
                safe=bool(safe),
            )
        )
        _log(f"Evicted {count} JLC ControlNet cache entr{'y' if count == 1 else 'ies'}.", verbose=verbose)
        return count
    except Exception as exc:
        _warn(f"JLC ControlNet cache eviction failed: {exc}")
        return 0


def _evict_all_jlc_cache(*, safe: bool, verbose: bool = True) -> int:
    """Explicit hammer: evict all JLC-managed resident cache entries."""

    if jlc_unload_all is None:
        if _JLC_CACHE_IMPORT_ERROR is not None:
            _warn(f"could not import JLC cache core: {_JLC_CACHE_IMPORT_ERROR}")
        else:
            _warn("JLC cache core unload_all() is unavailable.")
        return 0

    try:
        count = int(
            jlc_unload_all(
                include_keep=True,
                reason="stage_boundary_jlc_unload_all",
                safe=bool(safe),
            )
        )
        _log(f"Evicted {count} total JLC cache entr{'y' if count == 1 else 'ies'}.", verbose=verbose)
        return count
    except Exception as exc:
        _warn(f"JLC cache unload_all failed: {exc}")
        return 0


def _final_allocator_cleanup(*, safe: bool, verbose: bool = True) -> None:
    """Run backend-aware Comfy cleanup and JLC defensive CUDA cleanup."""

    gc.collect()
    _soft_empty_cache(verbose=verbose)

    if jlc_cuda_cleanup is not None:
        try:
            jlc_cuda_cleanup(
                reason="stage_boundary_final_allocator_cleanup",
                synchronize=True,
                safe=bool(safe),
            )
            _log(f"JLC allocator cleanup executed; safe={bool(safe)}.", verbose=verbose)
            return
        except Exception as exc:
            _warn(f"JLC allocator cleanup failed: {exc}")

    # Last-resort fallback only if the shared cache core could not be imported.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.synchronize()
        gc.collect()
        _log("Fallback torch CUDA allocator cleanup executed.", verbose=verbose)
    except Exception as exc:
        _warn(f"fallback CUDA allocator cleanup failed: {exc}")


class JLC_StageBoundaryVRAMCleanup:
    """Latent-triggered stage-boundary VRAM cleanup passthrough node."""

    FUNCTION = "cleanup"
    CATEGORY = "JLC/utils"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "unload_connected_model": ("BOOLEAN", {"default": True}),
                "evict_jlc_controlnet_cache": ("BOOLEAN", {"default": False}),
                "evict_all_jlc_model_cache": ("BOOLEAN", {"default": False}),
                "unload_all_comfy_models": ("BOOLEAN", {"default": False}),
                "clear_cuda_allocator": ("BOOLEAN", {"default": True}),
                "safe_cleanup": ("BOOLEAN", {"default": True}),
                "all_devices": ("BOOLEAN", {"default": False}),
                "verbose": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model": ("MODEL",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This node has intentional side effects, so force execution whenever it
        # sits on an active graph path instead of relying only on cached latents.
        return time.time()

    def cleanup(
        self,
        latent,
        unload_connected_model: bool = True,
        evict_jlc_controlnet_cache: bool = False,
        evict_all_jlc_model_cache: bool = False,
        unload_all_comfy_models: bool = False,
        clear_cuda_allocator: bool = True,
        safe_cleanup: bool = True,
        all_devices: bool = False,
        verbose: bool = True,
        model: Any = None,
    ):
        started = time.time()
        _log("Stage-boundary cleanup triggered by LATENT input.", verbose=verbose)

        # ComfyUI model residency cleanup.
        if bool(unload_all_comfy_models):
            _unload_all_comfy_models(verbose=verbose)
        elif bool(unload_connected_model):
            _unload_connected_model(model, all_devices=bool(all_devices), verbose=verbose)

        # JLC-owned resident cache cleanup. If the all-cache hammer is selected,
        # do not separately evict the ControlNet family first.
        if bool(evict_all_jlc_model_cache):
            _evict_all_jlc_cache(safe=bool(safe_cleanup), verbose=verbose)
        elif bool(evict_jlc_controlnet_cache):
            _evict_jlc_controlnet_cache(safe=bool(safe_cleanup), verbose=verbose)

        if bool(clear_cuda_allocator):
            _final_allocator_cleanup(safe=bool(safe_cleanup), verbose=verbose)
        else:
            gc.collect()
            _log("Allocator cleanup disabled; Python gc.collect() only.", verbose=verbose)

        elapsed = time.time() - started
        _log(f"Cleanup complete in {elapsed:.3f}s. Passing LATENT through.", verbose=verbose)
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "JLC_StageBoundaryVRAMCleanup": JLC_StageBoundaryVRAMCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_StageBoundaryVRAMCleanup": "\u2003JLC Stage Boundary VRAM Cleanup",
}
