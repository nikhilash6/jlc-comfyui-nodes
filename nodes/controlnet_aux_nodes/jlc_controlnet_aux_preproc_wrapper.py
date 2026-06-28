import importlib
from typing import Any, Dict, Iterable, List, Tuple

from ...jlc_custom_nodes_versions import JLC_CONTROLNET_AUX_VERSION

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
# - This wrapper is intentionally simple.
# - It exposes only preprocessors with:
#       IMAGE input
#       optional/shared resolution input
#       IMAGE-only output
# - Preprocessors with extra user parameters, model selectors, pose JSON,
#   masks, keypoints, dict-only outputs, or other special behavior are excluded.
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
    "Dynamic multi-slot wrapper for simple image-in/image-out preprocessors "
    "provided by Fannovel16's comfyui_controlnet_aux package. Only nodes with "
    "an IMAGE input, optional shared resolution input, and IMAGE-only output "
    "are exposed. Parameter-heavy preprocessors such as Canny, DWPose, OpenPose, "
    "and other special-output nodes should use their native nodes instead."
)

# ------------------------------------------------------------
# Dependency state
# ------------------------------------------------------------
AUX_AVAILABLE = False
AUX_IMPORT_ERROR = None
AUX_NODE_MAPPINGS: Dict[str, Any] = {}
AUX_SIMPLE_COMPATIBLE_NAMES: List[str] = []
AUX_SKIPPED_WHITELIST_NAMES: List[str] = []


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


def _is_simple_image_resolution_node(name: str, aux_class: Any) -> bool:
    """
    Strict compatibility gate for this generic wrapper.

    Accepted shape:
        required: image, and optionally resolution
        optional: absent, or resolution only
        returns: exactly one IMAGE output

    Anything requiring thresholds, detector toggles, model selectors, pose JSON,
    masks, keypoints, or other custom parameters is intentionally excluded.
    """
    if name in JLC_CNAUX_PREPROCESSOR_BLACKLIST:
        return False

    if not hasattr(aux_class, "INPUT_TYPES") or not hasattr(aux_class, "FUNCTION"):
        return False

    try:
        input_types = aux_class.INPUT_TYPES()
    except Exception:
        return False

    required = _as_name_set(input_types, "required")
    optional = _as_name_set(input_types, "optional")

    if "image" not in required:
        return False

    if not required.issubset({"image", "resolution"}):
        return False

    if not optional.issubset({"resolution"}):
        return False

    return_types = tuple(getattr(aux_class, "RETURN_TYPES", ()) or ())
    if return_types != ("IMAGE",):
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
    aux_pkg = importlib.import_module("comfyui_controlnet_aux")

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

    AUX_AVAILABLE = True

except Exception as e:
    AUX_IMPORT_ERROR = e


# ------------------------------------------------------------
# Helper: build argument payload for selected aux node
# ------------------------------------------------------------
def _build_aux_params(aux_class: Any, image: Any, resolution: int) -> Dict[str, Any]:
    """
    Build parameters for a simple-compatible aux node.

    Unlike the older wrapper draft, this intentionally does NOT auto-fill
    arbitrary primitive defaults. If a preprocessor needs thresholds, toggles,
    model selectors, masks, or other parameters, it is not a good fit for this
    generic simple wrapper and should be run with its native node.
    """
    combined = _get_combined_input_types(aux_class)
    params = {}

    for name in combined.keys():
        if name == "image":
            params[name] = image
        elif name == "resolution":
            params[name] = resolution
        else:
            raise RuntimeError(
                f"Aux preprocessor '{aux_class.__name__}' is not simple-wrapper "
                f"compatible because it requires parameter '{name}'."
            )

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
    if not AUX_AVAILABLE:
        selected = [name for name in _active_preprocessor_names(slot_count, kwargs) if name != "DISABLED"]
        if selected:
            return (
                "JLC Aux Preprocessor Wrapper requires Fannovel16's "
                "comfyui_controlnet_aux package. "
                f"Import/probe error: {AUX_IMPORT_ERROR}"
            )
        return True

    for name in _active_preprocessor_names(slot_count, kwargs):
        if name == "DISABLED":
            continue
        aux_class = AUX_NODE_MAPPINGS.get(name)
        if aux_class is None:
            return f"Preprocessor '{name}' was not found in comfyui_controlnet_aux."
        if not _is_simple_image_resolution_node(name, aux_class):
            return (
                f"Preprocessor '{name}' is not simple-wrapper compatible. Use its native "
                "ControlNet Auxiliary node instead."
            )

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