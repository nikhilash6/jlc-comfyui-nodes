"""
JLC Dynamic Aux Preprocessor Wrapper
------------------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for image generation
    pipelines, particularly:
        • Flux-based workflows
        • ControlNet composition and orchestration
        • LoRA experimentation
        • advanced inpainting / outpainting pipelines

- Upstream Dependency and Attribution
    The **JLC Dynamic Aux Preprocessor Wrapper** is a convenience wrapper around
    selected preprocessors provided by **Fannovel16's comfyui_controlnet_aux**
    package.

    The upstream package must be installed and importable for non-disabled
    preprocessor slots to execute.

    This node does not replace, supplant, or claim originality for the upstream
    ControlNet Auxiliary preprocessors. It exists only to provide a compact
    multi-slot interface for simple image-in/image-out preprocessors in JLC
    workflows.

    For preprocessors with thresholds, detector toggles, model selectors,
    pose/keypoint outputs, masks, segmentation payloads, optical flow, or other
    special behavior, use the native comfyui_controlnet_aux nodes instead.

- Node Purpose
    The wrapper exposes up to ten preprocessor slots for compact ControlNet Aux
    preprocessing.

    Supported preprocessor shape:
        • IMAGE input
        • optional/shared resolution input
        • IMAGE-only output
        • any additional parameters must have safe upstream defaults

    The node is intentionally conservative. It uses a curated include/exclude
    policy plus a default-aware compatibility gate before exposing upstream
    preprocessors.

- Dynamic Slot Design
    • Python predeclares ten preprocessor widgets and ten IMAGE outputs.
    • `slot_count` is authoritative.
    • Frontend JavaScript may hide widgets and output sockets above slot_count.
    • Hidden slot values remain serialized in workflow JSON.
    • Backend execution ignores hidden slots.
    • Hidden output slots return the source image as a passthrough placeholder
      so the static return shape remains stable.

- First-Use Model Note
    Some upstream ControlNet Aux preprocessors may download or load large
    auxiliary models the first time they are used. This behavior belongs to the
    upstream preprocessor package and model ecosystem, not to the JLC wrapper.

- Versioning
    Version is governed by `JLC_CONTROLNET_AUX_VERSION` from
    `jlc_custom_nodes_versions.py`.

- Attribution & License
  - JLC wrapper concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Built for use with:
    https://github.com/Fannovel16/comfyui_controlnet_aux

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ...jlc_custom_nodes_versions import JLC_CONTROLNET_AUX_VERSION

MANIFEST = {
    "name": "JLC Dynamic Aux Preprocessor Wrapper",
    "version": JLC_CONTROLNET_AUX_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Dynamic multi-slot convenience wrapper for IMAGE-to-IMAGE "
        "preprocessors provided by Fannovel16's comfyui_controlnet_aux package. "
        "Predeclares up to ten preprocessor slots and treats slot_count as "
        "authoritative while frontend JavaScript hides/shows rows and output "
        "sockets. Exposes preprocessors with IMAGE input, optional/shared "
        "resolution, IMAGE-only output, and defaultable extra parameters; "
        "special-output preprocessors should be used through their native "
        "ControlNet Aux nodes."
    ),
}

from .jlc_controlnet_aux_preproc_lists import (
    JLC_CNAUX_PREPROCESSOR_WHITELIST,
    JLC_CNAUX_PREPROCESSOR_BLACKLIST,
    JLC_CNAUX_AUTODISCOVER_SIMPLE_PREPROCESSORS,
)

# ------------------------------------------------------------
# JLC Dynamic Aux Preprocessor Wrapper
# ------------------------------------------------------------
# Multi-slot convenience wrapper for SIMPLE preprocessors provided by
# Fannovel16's comfyui_controlnet_aux package.
#
# Design rule:
# - This wrapper is intentionally compact.
# - It exposes preprocessors with:
#       IMAGE input
#       optional/shared resolution input
#       IMAGE-only output
#       defaultable extra parameters, filled from upstream defaults
# - Preprocessors with pose JSON, masks, keypoints, dict-only outputs, or other
#   special behavior are excluded.
#
# Dynamic slot semantics:
# - Python predeclares 10 output slots and 10 preprocessor widgets.
# - Frontend JS hides widgets and output sockets above slot_count.
# - Backend treats slot_count as authoritative.
# - Hidden slot widget values remain serialized in workflow JSON.
# ------------------------------------------------------------

MAX_SLOTS = 10
MAX_RESOLUTION = 16384

WRAPPER_TOOLTIP = (
    "Dynamic multi-slot wrapper for image-in/image-out preprocessors "
    "provided by Fannovel16's comfyui_controlnet_aux package. The node exposes "
    "preprocessors with IMAGE input, optional shared resolution, IMAGE-only output, "
    "and defaultable extra parameters. Special-output nodes should use their "
    "native ControlNet Aux nodes instead."
)

# ------------------------------------------------------------
# Dependency state
# ------------------------------------------------------------
AUX_AVAILABLE = False
AUX_IMPORT_ERROR = None
AUX_NODE_MAPPINGS: Dict[str, Any] = {}
AUX_SIMPLE_COMPATIBLE_NAMES: List[str] = []
AUX_SKIPPED_WHITELIST_NAMES: List[str] = []

# ------------------------------------------------------------
# Robust Fannovel16 custom-node import resolver
# ------------------------------------------------------------
def _find_custom_nodes_root() -> Path | None:
    """
    Locate ComfyUI's custom_nodes directory from this file.

    A normal Python import may fail while ComfyUI is loading custom nodes,
    even when Fannovel16/comfyui_controlnet_aux is correctly installed as a
    sibling custom node.  This resolver makes the sibling package importable
    without requiring the user to modify PYTHONPATH.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "custom_nodes":
            return parent
    return None


def _import_fannovel_aux_package():
    """
    Import Fannovel16's comfyui_controlnet_aux package from the active ComfyUI
    custom_nodes tree.

    The diagnostic command that succeeds for this case is equivalent to adding
    ComfyUI/custom_nodes to sys.path before importing comfyui_controlnet_aux.
    """
    errors = []

    # If ComfyUI already loaded the extension under the expected package name,
    # use that module directly.
    existing = sys.modules.get("comfyui_controlnet_aux")
    if existing is not None:
        return existing

    # First try the ordinary import in case custom_nodes is already on sys.path.
    try:
        return importlib.import_module("comfyui_controlnet_aux")
    except Exception as e:
        errors.append(f"plain import failed: {type(e).__name__}: {e}")

    # Then add the sibling custom_nodes root and try again.
    custom_nodes_root = _find_custom_nodes_root()
    if custom_nodes_root is not None:
        root_str = str(custom_nodes_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            return importlib.import_module("comfyui_controlnet_aux")
        except Exception as e:
            errors.append(f"custom_nodes import failed: {type(e).__name__}: {e}")

    raise RuntimeError("; ".join(errors) if errors else "unknown import failure")


def _unique_ordered(names: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _fallback_preprocessor_options() -> List[str]:
    """
    Fallback dropdown used when comfyui_controlnet_aux cannot be imported.

    This intentionally includes the static whitelist rather than collapsing to
    ["DISABLED"]. That avoids misleading ComfyUI validation errors such as a
    saved workflow value not being found in a one-item DISABLED-only combo list.
    Execution still raises the real aux import error if a non-disabled slot runs.
    """
    names = [
        name
        for name in JLC_CNAUX_PREPROCESSOR_WHITELIST
        if name != "DISABLED" and name not in JLC_CNAUX_PREPROCESSOR_BLACKLIST
    ]
    return ["DISABLED"] + _unique_ordered(names)


PREPROCESSOR_OPTIONS = _fallback_preprocessor_options()


# ------------------------------------------------------------
# Aux node inspection helpers
# ------------------------------------------------------------
def _as_name_set(input_types: Dict[str, Any], section: str) -> set:
    values = input_types.get(section, {}) or {}
    if not isinstance(values, dict):
        return set()
    return set(values.keys())


def _get_combined_input_types(aux_class: Any) -> Dict[str, Any]:
    input_types = aux_class.INPUT_TYPES()
    combined = {}
    combined.update(input_types.get("required", {}) or {})
    combined.update(input_types.get("optional", {}) or {})
    return combined


def _input_default_value(input_type: Any) -> tuple[bool, Any]:
    """
    Return (has_default, value) for an aux input spec.

    This mirrors the useful part of Fannovel16's AIO_Preprocessor behavior:
    non-image/non-resolution parameters may be auto-filled when the upstream
    node defines a default.  A few primitive types get safe fallbacks so older
    aux nodes without explicit defaults can still run when appropriate.
    """
    if isinstance(input_type, (list, tuple)):
        spec = input_type[0] if len(input_type) >= 1 else None
        meta = input_type[1] if len(input_type) >= 2 and isinstance(input_type[1], dict) else {}
    else:
        spec = input_type
        meta = {}

    if "default" in meta:
        return True, meta["default"]

    if spec == "INT":
        return True, 0
    if spec == "FLOAT":
        return True, 0.0
    if spec == "BOOLEAN":
        return True, False
    if spec == "STRING":
        return True, ""

    # For combo/list parameters, do not guess unless the upstream node supplied
    # an explicit default.  Many combos are model/mode selectors, and guessing
    # the first value can silently choose surprising behavior.
    return False, None


def _is_simple_image_resolution_node(name: str, aux_class: Any) -> bool:
    """
    Compatibility gate for this compact wrapper.

    Accepted shape:
        required/optional inputs include image
        resolution is passed from the shared widget when present
        other parameters must be defaultable from the upstream INPUT_TYPES spec
        returns exactly one IMAGE output

    This allows preprocessors such as DSINE, which are still image-to-image but
    expose defaulted controls like fov/iterations in the native node.
    """
    if name in JLC_CNAUX_PREPROCESSOR_BLACKLIST:
        return False

    if not hasattr(aux_class, "INPUT_TYPES") or not hasattr(aux_class, "FUNCTION"):
        return False

    try:
        input_types = aux_class.INPUT_TYPES()
    except Exception:
        return False

    required = input_types.get("required", {}) or {}
    optional = input_types.get("optional", {}) or {}
    if not isinstance(required, dict) or not isinstance(optional, dict):
        return False

    combined = {}
    combined.update(required)
    combined.update(optional)

    if "image" not in combined:
        return False

    return_types = tuple(getattr(aux_class, "RETURN_TYPES", ()) or ())
    if return_types != ("IMAGE",):
        return False

    for param_name, input_type in combined.items():
        if param_name in {"image", "resolution"}:
            continue
        has_default, _ = _input_default_value(input_type)
        if not has_default:
            return False

    return True

def _build_curated_preprocessor_options() -> List[str]:
    """
    Build final dropdown options.

    Ordering:
    1. DISABLED
    2. Whitelisted names, preserving local list order, if currently available
       and structurally simple-compatible
    3. Optionally, auto-discovered simple-compatible aux nodes not listed in
       the whitelist, sorted alphabetically
    """
    simple_names = []

    for name, aux_class in AUX_NODE_MAPPINGS.items():
        if _is_simple_image_resolution_node(name, aux_class):
            simple_names.append(name)

    simple_set = set(simple_names)

    curated = []
    for name in JLC_CNAUX_PREPROCESSOR_WHITELIST:
        if name == "DISABLED":
            continue
        if name in simple_set:
            curated.append(name)

    if JLC_CNAUX_AUTODISCOVER_SIMPLE_PREPROCESSORS:
        for name in sorted(simple_set):
            if name not in curated:
                curated.append(name)

    return ["DISABLED"] + _unique_ordered(curated)


# ------------------------------------------------------------
# Probe Fannovel dependency once at import time
# ------------------------------------------------------------
try:
    aux_pkg = _import_fannovel_aux_package()

    # Fannovel exposes AUX_NODE_MAPPINGS in current builds. NODE_CLASS_MAPPINGS
    # fallback makes this wrapper a little more resilient to package changes;
    # the strict structural gate still prevents non-preprocessor utility nodes
    # from entering the dropdown.
    AUX_NODE_MAPPINGS = (
        getattr(aux_pkg, "AUX_NODE_MAPPINGS", None)
        or getattr(aux_pkg, "NODE_CLASS_MAPPINGS", None)
        or {}
    )

    if not isinstance(AUX_NODE_MAPPINGS, dict) or not AUX_NODE_MAPPINGS:
        raise RuntimeError(
            "comfyui_controlnet_aux imported, but no AUX_NODE_MAPPINGS or "
            "NODE_CLASS_MAPPINGS dictionary was found."
        )

    PREPROCESSOR_OPTIONS = _build_curated_preprocessor_options()
    AUX_SIMPLE_COMPATIBLE_NAMES = [name for name in PREPROCESSOR_OPTIONS if name != "DISABLED"]

    AUX_SKIPPED_WHITELIST_NAMES = [
        name
        for name in JLC_CNAUX_PREPROCESSOR_WHITELIST
        if name != "DISABLED"
        and name not in AUX_SIMPLE_COMPATIBLE_NAMES
        and name not in JLC_CNAUX_PREPROCESSOR_BLACKLIST
    ]

    upstream_count = len(getattr(aux_pkg, "PREPROCESSOR_OPTIONS", []) or AUX_NODE_MAPPINGS)
    exposed_count = len(AUX_SIMPLE_COMPATIBLE_NAMES)

    print(
        f"[JLC Aux Wrapper] Loaded comfyui_controlnet_aux: "
        f"{upstream_count} upstream preprocessors, "
        f"{exposed_count} exposed simple preprocessors."
    )

    AUX_AVAILABLE = True

except Exception as e:
    AUX_IMPORT_ERROR = e


# ------------------------------------------------------------
# Helper: build argument payload for selected aux node
# ------------------------------------------------------------
def _build_aux_params(aux_class: Any, image: Any, resolution: int) -> Dict[str, Any]:
    """
    Build parameters for a default-aware image-to-image aux node.

    image is wired from the JLC node input, resolution is wired from the shared
    JLC widget, and any other parameters are filled from upstream defaults.
    """
    combined = _get_combined_input_types(aux_class)
    params = {}

    for name, input_type in combined.items():
        if name == "image":
            params[name] = image
        elif name == "resolution":
            params[name] = resolution
        else:
            has_default, default_value = _input_default_value(input_type)
            if not has_default:
                raise RuntimeError(
                    f"Aux preprocessor '{aux_class.__name__}' is not compact-wrapper "
                    f"compatible because parameter '{name}' has no safe default."
                )
            params[name] = default_value

    return params

def _extract_image_from_result(preprocessor_name: str, result: Any) -> Any:
    """
    Extract IMAGE from standard Comfy tuple return or a Comfy dict return.
    The structural gate should already exclude multi-output nodes, but this
    helper gives a clearer error if an aux node changes behavior upstream.
    """
    if isinstance(result, tuple) and len(result) > 0:
        return result[0]

    if isinstance(result, dict):
        payload = result.get("result")
        if isinstance(payload, tuple) and len(payload) > 0:
            return payload[0]
        if isinstance(payload, list) and len(payload) > 0:
            return payload[0]

    raise RuntimeError(
        f"Preprocessor '{preprocessor_name}' returned an unsupported result shape: "
        f"{type(result)}"
    )


# ------------------------------------------------------------
# Helper: execute one selected preprocessor
# ------------------------------------------------------------
def _run_preprocessor(preprocessor_name: str, image: Any, resolution: int) -> Any:
    """
    Execute a single selected preprocessor.

    DISABLED is a no-op passthrough in this wrapper.
    """
    if preprocessor_name == "DISABLED":
        return image

    if not AUX_AVAILABLE:
        raise RuntimeError(
            "JLC Aux Preprocessor Wrapper requires Fannovel16's "
            "comfyui_controlnet_aux package to be installed and importable.\n"
            f"Import/probe error: {AUX_IMPORT_ERROR}"
        )

    if preprocessor_name not in AUX_NODE_MAPPINGS:
        raise RuntimeError(
            f"Preprocessor '{preprocessor_name}' was not found in "
            "comfyui_controlnet_aux AUX_NODE_MAPPINGS."
        )

    aux_class = AUX_NODE_MAPPINGS[preprocessor_name]

    if not _is_simple_image_resolution_node(preprocessor_name, aux_class):
        raise RuntimeError(
            f"Preprocessor '{preprocessor_name}' is not simple-wrapper compatible. "
            "Use its native ControlNet Auxiliary node instead. This dynamic wrapper "
            "only supports IMAGE input, optional shared resolution, and IMAGE-only output."
        )

    params = _build_aux_params(aux_class, image=image, resolution=resolution)
    result = getattr(aux_class(), aux_class.FUNCTION)(**params)
    return _extract_image_from_result(preprocessor_name, result)


def _coerce_slot_count(value: Any, default: int = 1) -> int:
    try:
        count = int(value)
    except Exception:
        count = default
    return max(1, min(MAX_SLOTS, count))


def _active_preprocessor_names(slot_count: int, kwargs: Dict[str, Any]) -> List[str]:
    names = []
    for i in range(1, slot_count + 1):
        suffix = f"{i:02d}"
        names.append(kwargs.get(f"preprocessor_{suffix}", "DISABLED"))
    return names


def _validate_active_preprocessors(slot_count: int, kwargs: Dict[str, Any]) -> Any:
    # Keep prompt validation permissive.  ComfyUI repeats custom validation
    # failures once per widget, which creates noisy logs and prevents the graph
    # from reaching execution.  Missing packages, stale saved dropdown values,
    # and incompatible preprocessors are handled in _run_preprocessor, where the
    # user gets one actionable runtime error instead of a dozen validation lines.
    return True

# ------------------------------------------------------------
# Dynamic wrapper class
# ------------------------------------------------------------
class JLC_DynamicAuxPreprocessorWrapper:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "slot_count": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": MAX_SLOTS,
                    "step": 1,
                    "tooltip": "Number of visible/executed aux preprocessor slots.",
                },
            ),
            "resolution": (
                "INT",
                {
                    "default": 512,
                    "min": 64,
                    "max": MAX_RESOLUTION,
                    "step": 64,
                    "tooltip": WRAPPER_TOOLTIP,
                },
            ),
        }

        for i in range(1, MAX_SLOTS + 1):
            suffix = f"{i:02d}"
            required[f"preprocessor_{suffix}"] = (
                PREPROCESSOR_OPTIONS,
                {
                    "default": "DISABLED",
                    "tooltip": WRAPPER_TOOLTIP,
                },
            )

        return {"required": required}

    RETURN_TYPES = tuple(["IMAGE"] * MAX_SLOTS)
    RETURN_NAMES = tuple([f"image_{i:02d}" for i in range(1, MAX_SLOTS + 1)])
    FUNCTION = "execute"
    CATEGORY = "JLC/ControlNet"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        slot_count = _coerce_slot_count(kwargs.get("slot_count", 1))
        return _validate_active_preprocessors(slot_count, kwargs)

    def execute(self, image, slot_count=1, resolution=512, **kwargs):
        count = _coerce_slot_count(slot_count)
        outputs = []

        for i in range(1, MAX_SLOTS + 1):
            suffix = f"{i:02d}"

            if i > count:
                # Hidden slots are intentionally ignored by backend execution.
                # Return passthrough image to keep Comfy's static return arity stable.
                outputs.append(image)
                continue

            preprocessor_name = kwargs.get(f"preprocessor_{suffix}", "DISABLED")
            outputs.append(
                _run_preprocessor(
                    preprocessor_name=preprocessor_name,
                    image=image,
                    resolution=resolution,
                )
            )

        return tuple(outputs)


# ------------------------------------------------------------
# Register dynamic variant
# ------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "JLC_DynamicAuxPreprocessorWrapper": JLC_DynamicAuxPreprocessorWrapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_DynamicAuxPreprocessorWrapper": "\u2003JLC Dynamic Aux Preprocessor Wrapper",
}