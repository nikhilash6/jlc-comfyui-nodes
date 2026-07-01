"""
JLC ControlNet Apply
--------------------

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
  - The **JLC ControlNet Apply** node applies one ControlNet to both
    positive and negative conditioning streams while preserving native
    ComfyUI ControlNet chaining semantics.

  - This node is intentionally simple and native-facing:
        • the hint image is a required input and is validated by ComfyUI;
        • the incoming ControlNet is copied before conditioning state is applied;
        • previously attached ControlNets are preserved through
          `set_previous_controlnet(prev_cnet)`;
        • disabled or zero-strength operation is a pass-through.

  - In the larger JLC non-recursive ControlNet workflow, this node is useful
    as a conventional chain-building stage.  A downstream Composition node
    may later extract the native chain, detach it, and replace recursive
    evaluation with weighted non-recursive fusion.

  - The node also passes the ControlNet and VAE forward so that chained
    workflows remain tidy and do not require repeated loader wiring.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Adapted from the **ControlNetApply** node in the core **ComfyUI**
    project:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

MANIFEST = {
    "name": "JLC ControlNet Apply",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Native-style ControlNet apply node for chained workflows. Applies one "
        "ControlNet to positive and negative conditioning, preserves existing "
        "previous_controlnet chains, supports pass-through disabling, and passes "
        "ControlNet/VAE objects forward for tidy multi-node pipelines."
    ),
}


class JLC_ControlNetApply:
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "vae": ("VAE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONTROL_NET", "VAE")
    RETURN_NAMES = ("positive", "negative", "control_net", "vae")

    def apply_controlnet(
        self,
        enabled,
        positive,
        negative,
        control_net,
        image,
        strength,
        start_percent,
        end_percent,
        vae,
        extra_concat=None,
    ):
        if (not enabled) or strength == 0:
            return (positive, negative, control_net, vae)

        if extra_concat is None:
            extra_concat = []

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
                            extra_concat=extra_concat,
                        )
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                c.append([t[0], d])

            out.append(c)

        return (out[0], out[1], control_net, vae)


NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetApply": JLC_ControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetApply": "JLC ControlNet Apply",
}
