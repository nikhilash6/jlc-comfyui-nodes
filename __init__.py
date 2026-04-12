"""
JLC ComfyUI Custom Nodes
------------------------

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

- This module registers all nodes in the JLC node collection.

- ComfyUI loads the package located in `custom_nodes/<repo_name>/` and reads
  the following mappings:

    NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS

- Each entry maps an internal class name to the class implementing the node.

- Actual node implementations live in the `nodes/` subdirectory to keep the
  repository organized as the number of nodes grows.

- Author: J. L. Córdova
  License: MIT
"""

# __init__.py  (JLC-Tests-comfyui)

import os
import sys
from server import PromptServer  # used for static route mounting

from .nodes.jlc_padded_image import JLC_PaddedImage
from .nodes.jlc_padded_latent import JLC_PaddedLatent
from .nodes.jlc_controlnet_apply import JLC_ControlNetApply
from .nodes.jlc_controlnet_apply_advanced import JLC_ControlNetApplyAdvanced
from .nodes.jlc_lora_loader_ten_stack import JLC_LoraLoaderTenStack
from .nodes.jlc_lora_loader_block_weight_two import JLC_LoraLoaderBlockWeightTwo
from .nodes.jlc_seed_generator import JLC_SeedGenerator
from .nodes.jlc_controlnet_composition import JLC_ControlNetComposition
# from .nodes.jlc_controlnet_orchestrator import JLC_ControlNetOrchestrator
# from .nodes.jlc_cuda_cleanup import JLC_CudaCleanup


print("🧠 JLC Nodes Loaded:")
# print("🔥 INIT LOADED:", __file__)

# for name, mod in sys.modules.items():
#     if "jlc" in name.lower():
#         print("🧠 JLC MODULE:", name, getattr(mod, "__file__", None))

NODE_CLASS_MAPPINGS = {
    "JLC_PaddedImage": JLC_PaddedImage,
    "JLC_PaddedLatent": JLC_PaddedLatent,
    "JLC_ControlNetApply": JLC_ControlNetApply,
    "JLC_ControlNetApplyAdvanced": JLC_ControlNetApplyAdvanced,
    "JLC_LoraLoaderTenStack": JLC_LoraLoaderTenStack,
    "JLC_LoraLoaderBlockWeightTwo": JLC_LoraLoaderBlockWeightTwo,
    "JLC_SeedGenerator": JLC_SeedGenerator,
    "JLC_ControlNetComposition": JLC_ControlNetComposition,
    # "JLC_ControlNetOrchestrator": JLC_ControlNetOrchestrator,
    # "JLC_CudaCleanup": JLC_CudaCleanup,
}

# Keep \u2003 (em-dash) leading space in names to avoid logo overlap;
NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_PaddedImage": "\u2003JLC Padded Image",
    "JLC_PaddedLatent": "\u2003JLC Padded Latent",
    "JLC_ControlNetApply": "\u2003JLC ControlNet Apply",
    "JLC_ControlNetApplyAdvanced": "\u2003JLC ControlNet Apply (Advanced)",
    "JLC_LoraLoaderTenStack": "\u2003JLC 10-LoRA Loader",
    "JLC_LoraLoaderBlockWeightTwo": "\u2003JLC 2-LoRA Loader - Block Weight",
    "JLC_SeedGenerator": "\u2003JLC Seed Generator",
    "JLC_ControlNetComposition": "\u2003JLC ControlNet Composition",
    # "JLC_ControlNetOrchestrator": "\u2003JLC ControlNet Orchestrator",
    # "JLC_CudaCleanup": "\u2003JLC_CudaCleanup",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Path to web folder

WEB_DIRECTORY = "web"

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Mount it into ComfyUI frontend
ps = PromptServer.instance

if os.path.exists(WEB_DIR):
    ps.app.router.add_static(
        "/extensions/JLC-ComfyUI-nodes",
        WEB_DIR
    )