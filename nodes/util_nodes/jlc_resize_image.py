"""
JLC Resize Image
----------------

- JLC ComfyUI Nodes Collection
- Tensor-based resize utility node intended for mid-workflow use.
- Accepts an IMAGE input, preserves aspect ratio using the same resize math
  as JLC Load & Resize Image, and preserves NONE passthrough semantics for
  workflows where upstream nodes deliberately disable slots by returning None.

Behavior:
- Valid IMAGE input -> resize image and return aligned all-zero MASK.
- Runtime None input -> passthrough None for IMAGE and MASK, with width/height 0.

Notes:
- This node intentionally has no file-loader widget and no internal preview.
- MASK output is retained for possible future workflow use, but because this
  node receives only IMAGE, the returned MASK is currently an all-zero mask
  aligned to the resized image dimensions.
"""

from __future__ import annotations

import torch
from nodes import MAX_RESOLUTION

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION
from .jlc_load_and_resize_image import (
    RESIZE_MODES,
    SCALE_METHODS,
    calculate_target_dimensions,
    _resize_image,
    _resize_mask,
)


MANIFEST = {
    "name": "JLC Resize Image",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Resize an incoming IMAGE tensor with aspect-ratio-preserving math, "
        "final divisible-by alignment, and deliberate None passthrough for "
        "dynamic workflow branches."
    ),
}


class JLC_ResizeImage:
    FUNCTION = "resize_image"
    CATEGORY = "utils/image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    OUTPUT_NODE = False

    DESCRIPTION = (
        "Resize an incoming IMAGE tensor with aspect-ratio-preserving math and "
        "final divisible-by alignment. If the connected input evaluates to None, "
        "the node deliberately passes None through so downstream ControlNet / "
        "Reference nodes can handle disabled-slot behavior correctly."
    )
    SEARCH_ALIASES = [
        "resize image",
        "jlc resize",
        "aspect ratio resize",
        "resize image tensor",
        "none passthrough resize",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Incoming IMAGE tensor to resize. If the upstream "
                            "connected node deliberately outputs None at runtime, "
                            "this node passes None through."
                        ),
                    },
                ),
                "resize_by": (
                    RESIZE_MODES,
                    {
                        "default": "scale longer dimension",
                        "tooltip": (
                            "Aspect-ratio-preserving resize policy."
                        ),
                    },
                ),
                "multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale factor; 2.0 doubles both dimensions.",
                    },
                ),
                "longer_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": "Target size of the source image's longer edge.",
                    },
                ),
                "shorter_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": "Target size of the source image's shorter edge.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": (
                            "Target width; height is calculated from aspect ratio."
                        ),
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": (
                            "Target height; width is calculated from aspect ratio."
                        ),
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 64.0,
                        "step": 0.01,
                        "tooltip": (
                            "Target total megapixels using 1024×1024 per megapixel."
                        ),
                    },
                ),
                "scale_method": (
                    SCALE_METHODS,
                    {
                        "default": "area",
                        "tooltip": (
                            "Interpolation method. Area is generally strong for "
                            "downscaling; Lanczos is often useful for upscaling."
                        ),
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "After aspect-ratio calculation, round width and "
                            "height down to this multiple. Use 1 to disable."
                        ),
                    },
                ),
            }
        }

    def resize_image(
        self,
        image,
        resize_by,
        multiplier,
        longer_size,
        shorter_size,
        width,
        height,
        megapixels,
        scale_method,
        divisible_by,
    ):
        # Deliberate runtime NONE passthrough for dynamic workflows.
        if image is None:
            return (None, None, 0, 0)

        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(
                "JLC Resize Image expected an IMAGE tensor in BHWC layout "
                "or a deliberate runtime None."
            )

        source_height = int(image.shape[1])
        source_width = int(image.shape[2])

        target_width, target_height = calculate_target_dimensions(
            source_width,
            source_height,
            resize_by,
            multiplier=multiplier,
            longer_size=longer_size,
            shorter_size=shorter_size,
            width=width,
            height=height,
            megapixels=megapixels,
            divisible_by=divisible_by,
        )

        resized_image = _resize_image(
            image,
            target_width,
            target_height,
            scale_method,
        )

        # No incoming MASK exists on this node, so return an aligned zero mask.
        resized_mask = _resize_mask(
            None,
            image_batch=int(resized_image.shape[0]),
            target_width=target_width,
            target_height=target_height,
            scale_method=scale_method,
            dtype=resized_image.dtype,
        )

        return (
            resized_image,
            resized_mask,
            int(target_width),
            int(target_height),
        )