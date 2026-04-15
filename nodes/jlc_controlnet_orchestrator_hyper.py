"""
JLC ControlNet Orchestrator (Hyper - Flux)

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
    - The **JLC ControlNet Orchestrator (Hyper-Streaming)** introduces
      an experimental ControlNet execution architecture that departs
      from both ComfyUI’s native chained (`previous_controlnet`) model
      and traditional post-hoc composition strategies.

    - Instead of recursive chaining or deferred aggregation, this node:
            • Accepts multiple ControlNet inputs (slot-based)
            • Creates isolated instances via `.copy()` (no mutation)
            • Applies independent conditioning per slot
            • Executes each ControlNet against the same latent input
            • Performs **streaming fusion at emission time**

    - Fusion is defined as:
            combined = Σ (w_i · C_i(x))

      where:
            • C_i(x) = independent ControlNet outputs
            • w_i = user-defined weights (supports negative values)

    - Key innovation:
        - ControlNet outputs are fused **during model execution**
          via a streaming hook into Flux (`forward_orig`), rather than
          after execution via Python-level aggregation.

    - This approach:
            • Eliminates recursive chaining and state inheritance
            • Eliminates post-hoc ControlNet merging overhead
            • Preserves full ControlNet independence
            • Reduces Python-side overhead in multi-ControlNet workflows
            • Enables near-native performance with multiple ControlNets

    - Performance characteristics:
        - Achieves parity or improvement over prior composition methods
        - Demonstrates reduced overhead in multi-ControlNet scenarios
        - Exhibits sensitivity to GPU memory allocation behavior
          (cold-start OOM vs steady-state performance)

    - Philosophical deviation:
        - ControlNets are treated as **independent operators**
          whose interaction occurs **only at the level of emitted
          control signals**, not through chained execution.

    - ⚠️ Experimental Code:
        - This node introduces a novel execution paradigm and relies on
          internal model patching (Flux monkey patch).
        - Performance and memory behavior may vary depending on
          resolution, ControlNet count, and GPU characteristics.
        - Intended for advanced users and controlled testing scenarios.

    - ⚠️ Model Compatibility:
        - Designed and validated for Flux1-based workflows.
        - The implementation hooks into ComfyUI's unified ControlNetFlux
          execution path and may work with Flux2 and quantized variants.
        - However, compatibility with all Flux model variants has not been
          tested and may vary depending on backend and precision.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

MANIFEST = {
    "name": "JLC ControlNet Orchestrator (Hyper-Streaming)",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": (
        "Node implementing a novel Hyper-Streaming ControlNet execution architecture. "
        "Performs emission-time fusion of multiple ControlNets via a Flux hook, eliminating "
        "recursive chaining and post-hoc aggregation. Enables near-native performance and "
        "deterministic multi-ControlNet behavior with reduced Python overhead."
    ),
}


import torch
import math

DEBUG = True

import folder_paths
import comfy.controlnet
import comfy.utils
import comfy.model_management
import comfy.model_base
import comfy.ldm.flux.controlnet as comfy_flux_controlnet

# ------------------------------------------------------------
# 🪝 Monkey Patch — Flux streaming fusion hook
# ------------------------------------------------------------
_FLUX_STREAM_PATCHED = False
_ORIGINAL_FLUX_FORWARD = None
_ORIGINAL_FLUX_FORWARD_ORIG = None

def install_flux_stream_fuse_patch():
    global _FLUX_STREAM_PATCHED, _ORIGINAL_FLUX_FORWARD, _ORIGINAL_FLUX_FORWARD_ORIG

    if _FLUX_STREAM_PATCHED:
        return

    FluxClass = comfy_flux_controlnet.ControlNetFlux
    _ORIGINAL_FLUX_FORWARD = FluxClass.forward
    _ORIGINAL_FLUX_FORWARD_ORIG = FluxClass.forward_orig

    def patched_forward_orig(
        self,
        img,
        img_ids,
        controlnet_cond,
        txt,
        txt_ids,
        timesteps,
        y,
        guidance=None,
        control_type=None,
        stream_fuse=None,
    ):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        else:
            y = y[:, :self.params.vec_in_dim]

        img = self.img_in(img)

        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond
        vec = self.time_in(comfy_flux_controlnet.timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            vec = vec + self.guidance_in(comfy_flux_controlnet.timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        if self.controlnet_mode_embedder is not None and len(control_type) > 0:
            control_cond = self.controlnet_mode_embedder(
                torch.tensor(control_type, device=img.device),
                out_dtype=img.dtype,
            ).unsqueeze(0).repeat((txt.shape[0], 1, 1))
            txt = torch.cat([control_cond, txt], dim=1)
            txt_ids = torch.cat([txt_ids[:, :1], txt_ids], dim=1)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if stream_fuse is None:
            controlnet_double = ()
            for i in range(len(self.double_blocks)):
                img, txt = self.double_blocks[i](img=img, txt=txt, vec=vec, pe=pe)
                controlnet_double = controlnet_double + (self.controlnet_blocks[i](img),)

            img = torch.cat((txt, img), 1)

            controlnet_single = ()
            for i in range(len(self.single_blocks)):
                img = self.single_blocks[i](img, vec=vec, pe=pe)
                controlnet_single = controlnet_single + (self.controlnet_single_blocks[i](img[:, txt.shape[1]:, ...]),)

            repeat = math.ceil(self.main_model_double / len(controlnet_double))
            if self.latent_input:
                out_input = ()
                for x in controlnet_double:
                    out_input += (x,) * repeat
            else:
                out_input = controlnet_double * repeat

            out = {"input": out_input[:self.main_model_double]}
            if len(controlnet_single) > 0:
                repeat = math.ceil(self.main_model_single / len(controlnet_single))
                out_output = ()
                if self.latent_input:
                    for x in controlnet_single:
                        out_output += (x,) * repeat
                else:
                    out_output = controlnet_single * repeat
                out["output"] = out_output[:self.main_model_single]

            # ------------------------------------------------------------
            # 🟢 Original path (no streaming)
            # ------------------------------------------------------------
            return out

        controlnet_double = []
        for i in range(len(self.double_blocks)):
            img, txt = self.double_blocks[i](img=img, txt=txt, vec=vec, pe=pe)
            controlnet_double.append(self.controlnet_blocks[i](img))

        double_repeat = math.ceil(self.main_model_double / len(controlnet_double))
        emitted = 0
        if self.latent_input:
            for x in controlnet_double:
                for _ in range(double_repeat):
                    if emitted >= self.main_model_double:
                        break
                    stream_fuse("input", emitted, x)
                    emitted += 1
        else:
            for idx in range(self.main_model_double):
                x = controlnet_double[idx % len(controlnet_double)]
                stream_fuse("input", idx, x)

        img = torch.cat((txt, img), 1)

        controlnet_single = []
        for i in range(len(self.single_blocks)):
            img = self.single_blocks[i](img, vec=vec, pe=pe)
            controlnet_single.append(self.controlnet_single_blocks[i](img[:, txt.shape[1]:, ...]))

        if len(controlnet_single) > 0:
            single_repeat = math.ceil(self.main_model_single / len(controlnet_single))
            emitted = 0
            if self.latent_input:
                for x in controlnet_single:
                    for _ in range(single_repeat):
                        if emitted >= self.main_model_single:
                            break
                        stream_fuse("output", emitted, x)
                        emitted += 1
            else:
                for idx in range(self.main_model_single):
                    x = controlnet_single[idx % len(controlnet_single)]
                    stream_fuse("output", idx, x)

        return None

    def patched_forward(self, x, timesteps, context, y=None, guidance=None, hint=None, **kwargs):
        patch_size = 2
        if self.latent_input:
            hint = comfy_flux_controlnet.comfy.ldm.common_dit.pad_to_patch_size(hint, (patch_size, patch_size))
        elif self.mistoline:
            hint = hint * 2.0 - 1.0
            hint = self.input_cond_block(hint)
        else:
            hint = hint * 2.0 - 1.0
            hint = self.input_hint_block(hint)

        hint = comfy_flux_controlnet.rearrange(
            hint,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=patch_size,
            pw=patch_size,
        )

        bs, c, h, w = x.shape
        x = comfy_flux_controlnet.comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = comfy_flux_controlnet.rearrange(
            x,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=patch_size,
            pw=patch_size,
        )

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = comfy_flux_controlnet.repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        return self.forward_orig(
            img,
            img_ids,
            hint,
            context,
            txt_ids,
            timesteps,
            y,
            guidance,
            control_type=kwargs.get("control_type", []),
            stream_fuse=kwargs.get("stream_fuse", None),
        )

    FluxClass.forward_orig = patched_forward_orig
    FluxClass.forward = patched_forward
    _FLUX_STREAM_PATCHED = True

install_flux_stream_fuse_patch()

# ------------------------------------------------------------
# 🔒 Global ControlNet Cache (max 3 expected, no eviction)
# ------------------------------------------------------------
GLOBAL_CONTROLNET_CACHE = {}

# ------------------------------------------------------------
# 🧠 Core Fusion Wrapper: Novel Hyper Streaming Approach
# ------------------------------------------------------------
class JLC_ComposedControlNet:
    def __init__(self, controlnets, weights):
        self.controlnets = controlnets
        self.weights = weights
        self.previous_controlnet = None
        self.extra_hooks = None

    def _make_stream_fuser(self, cnet, weight, combined):
        main_input = getattr(cnet.control_model, "main_model_double", 0)
        main_output = getattr(cnet.control_model, "main_model_single", 0)

        if "input" not in combined:
            combined["input"] = [None] * main_input
        if "output" not in combined:
            combined["output"] = [None] * main_output

        input_list = combined["input"]
        output_list = combined["output"]

        main_counts = {
            "input": main_input,
            "output": main_output,
        }

        def stream_fuse(key, index, raw_x):
            if raw_x is None:
                return

            combined_list = input_list if key == "input" else output_list

            x = raw_x
            if cnet.global_average_pooling:
                x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(
                    1, 1, x.shape[2], x.shape[3]
                )

            if cnet.strength_type == comfy.controlnet.StrengthType.CONSTANT:
                if cnet.strength == 1.0:
                    x_scaled = x
                elif cnet.strength == 0.0:
                    return  # 🔥 skip entirely
                else:
                    x_scaled = x * cnet.strength
            elif cnet.strength_type == comfy.controlnet.StrengthType.LINEAR_UP:
                total = main_counts[key] or (index + 1)
                factor = cnet.strength ** float(total - index)
                if factor == 1.0:
                    x_scaled = x
                else:
                    x_scaled = x * factor
            else:
                x_scaled = x

            dst = combined_list[index]

            if dst is None:
                if weight == 1.0:
                    # 🔥 skip clone — take ownership directly
                    combined_list[index] = x_scaled
                else:
                    owned = x_scaled.clone()
                    owned.mul_(weight)
                    combined_list[index] = owned
            else:
                dst.add_(x_scaled, alpha=weight)

        return stream_fuse

    def _run_controlnet_no_merge(self, cnet, x_noisy, t, cond, batched_number, stream_fuse=None):
        # ------------------------------------------------------------
        # Mirror Comfy's ControlNet.get_control() setup path
        # but STOP before control_merge(...)
        # ------------------------------------------------------------
        if cnet.timestep_range is not None:
            if t[0] > cnet.timestep_range[0] or t[0] < cnet.timestep_range[1]:
                return None

        dtype = cnet.control_model.dtype
        if cnet.manual_cast_dtype is not None:
            dtype = cnet.manual_cast_dtype

        if (
            cnet.cond_hint is None
            or x_noisy.shape[2] * cnet.compression_ratio != cnet.cond_hint.shape[2]
            or x_noisy.shape[3] * cnet.compression_ratio != cnet.cond_hint.shape[3]
        ):
            if cnet.cond_hint is not None:
                del cnet.cond_hint

            cnet.cond_hint = None
            compression_ratio = cnet.compression_ratio

            if cnet.vae is not None:
                compression_ratio *= cnet.vae.spacial_compression_encode()
            else:
                if cnet.latent_format is not None:
                    raise ValueError(
                        "This Controlnet needs a VAE but none was provided, "
                        "please use a ControlNetApply node with a VAE input and connect it."
                    )

            cnet.cond_hint = comfy.utils.common_upscale(
                cnet.cond_hint_original,
                x_noisy.shape[-1] * compression_ratio,
                x_noisy.shape[-2] * compression_ratio,
                cnet.upscale_algorithm,
                "center",
            )

            cnet.cond_hint = cnet.preprocess_image(cnet.cond_hint)

            if cnet.vae is not None:
                loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
                cnet.cond_hint = cnet.vae.encode(cnet.cond_hint.movedim(1, -1))
                comfy.model_management.load_models_gpu(loaded_models)

            if cnet.latent_format is not None:
                cnet.cond_hint = cnet.latent_format.process_in(cnet.cond_hint)

            if len(cnet.extra_concat_orig) > 0:
                to_concat = []
                for c in cnet.extra_concat_orig:
                    c = c.to(cnet.cond_hint.device)
                    c = comfy.utils.common_upscale(
                        c,
                        cnet.cond_hint.shape[-1],
                        cnet.cond_hint.shape[-2],
                        cnet.upscale_algorithm,
                        "center",
                    )
                    if c.ndim < cnet.cond_hint.ndim:
                        c = c.unsqueeze(2)
                        c = comfy.utils.repeat_to_batch_size(c, cnet.cond_hint.shape[2], dim=2)
                    to_concat.append(comfy.utils.repeat_to_batch_size(c, cnet.cond_hint.shape[0]))

                cnet.cond_hint = torch.cat([cnet.cond_hint] + to_concat, dim=1)

            cnet.cond_hint = cnet.cond_hint.to(device=x_noisy.device, dtype=dtype)

        if x_noisy.shape[0] != cnet.cond_hint.shape[0]:
            cnet.cond_hint = comfy.controlnet.broadcast_image_to(
                cnet.cond_hint,
                x_noisy.shape[0],
                batched_number,
            )

        context = cond.get("crossattn_controlnet", cond["c_crossattn"])

        extra = cnet.extra_args.copy()
        for c in cnet.extra_conds:
            temp = cond.get(c, None)
            if temp is not None:
                extra[c] = comfy.model_base.convert_tensor(temp, dtype, x_noisy.device)

        if stream_fuse is not None and isinstance(cnet.control_model, comfy_flux_controlnet.ControlNetFlux):
            extra["stream_fuse"] = stream_fuse

        timestep = cnet.model_sampling_current.timestep(t)
        x_input = cnet.model_sampling_current.calculate_input(t, x_noisy)

        control = cnet.control_model(
            x=x_input.to(dtype),
            hint=cnet.cond_hint,
            timesteps=timestep.to(dtype),
            context=comfy.model_management.cast_to_device(context, x_noisy.device, dtype),
            **extra,
        )

        return control

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):

        combined = None

        for idx, (cnet, w) in enumerate(zip(self.controlnets, self.weights)):

            if cnet is None or w == 0:
                continue

            if combined is None:
                combined = {}

            stream_fuse = None
            use_stream_fuse = isinstance(cnet.control_model, comfy_flux_controlnet.ControlNetFlux)
            if use_stream_fuse:
                stream_fuse = self._make_stream_fuser(cnet, w, combined)

            control = self._run_controlnet_no_merge(
                cnet,
                x_noisy,
                t,
                cond,
                batched_number,
                stream_fuse=stream_fuse,
            )

            if control is None:
                if use_stream_fuse:
                    continue
                else:
                    continue

            if use_stream_fuse:
                del control
                continue

            for key, control_output in control.items():

                if key not in combined:
                    combined[key] = [None] * len(control_output)

                combined_list = combined[key]

                if len(combined_list) < len(control_output):
                    combined_list.extend([None] * (len(control_output) - len(combined_list)))

                for i in range(len(control_output)):
                    raw_x = control_output[i]

                    if raw_x is None:
                        continue

                    x = raw_x

                    if cnet.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(
                            1, 1, x.shape[2], x.shape[3]
                        )
                        # NOTE:
                        # This assignment appears unused, but removing it alters execution timing
                        # and can significantly impact CUDA memory allocation behavior, leading
                        # to performance regressions. Do not remove.
                        raw_key = x
                    else:
                        # NOTE:
                        # This assignment appears unused, but removing it alters execution timing
                        # and can significantly impact CUDA memory allocation behavior, leading
                        # to performance regressions. Do not remove.
                        raw_key = x
                        
                    if cnet.strength_type == comfy.controlnet.StrengthType.CONSTANT:
                        if cnet.strength == 1.0:
                            x_scaled = x
                        else:
                            x_scaled = x * cnet.strength
                    elif cnet.strength_type == comfy.controlnet.StrengthType.LINEAR_UP:
                        factor = cnet.strength ** float(len(control_output) - i)
                        if factor == 1.0:
                            x_scaled = x
                        else:
                            x_scaled = x * factor
                    else:
                        x_scaled = x

                    dst = combined_list[i]

                    if dst is None:
                        owned = x_scaled.clone()
                        if w != 1.0:
                            owned.mul_(w)
                        combined_list[i] = owned
                    else:
                        dst.add_(x_scaled, alpha=w)

            del control

        # ------------------------------------------------------------
        # Final normalization for Flux compatibility
        # Flux main model expects BOTH "input" and "output"
        # to be iterable sequences, never None.
        # ------------------------------------------------------------
        if combined is None:
            combined = {}

        if "input" not in combined or combined["input"] is None:
            combined["input"] = [None] * 19   # Flux main_model_double

        if "output" not in combined or combined["output"] is None:
            combined["output"] = [None] * 38  # Flux main_model_single

        return combined


# ------------------------------------------------------------
# 🧠 KSampler Compatibility — ControlNet Interface Passthrough
# ------------------------------------------------------------
# These methods forward required ControlNet interface calls to all
# underlying ControlNet instances, preserving full compatibility
# with ComfyUI's execution pipeline (hooks, model loading,
# memory estimation, and lifecycle management).

    def get_extra_hooks(self):
        hooks = []
        for cnet in self.controlnets:
            if hasattr(cnet, "get_extra_hooks"):
                hooks += cnet.get_extra_hooks()
        return hooks


    def get_models(self):
        models = []
        for cnet in self.controlnets:
            if hasattr(cnet, "get_models"):
                models += cnet.get_models()
        return models


    def inference_memory_requirements(self, dtype):
        max_req = 0
        for cnet in self.controlnets:
            if cnet is None:
                continue
            if hasattr(cnet, "inference_memory_requirements"):
                req = cnet.inference_memory_requirements(dtype)
                if req is not None:
                    max_req = max(max_req, req)
        return max_req


    def pre_run(self, model, percent_to_timestep_function):
        for cnet in self.controlnets:
            if hasattr(cnet, "pre_run"):
                cnet.pre_run(model, percent_to_timestep_function)


    def cleanup(self):
        for cnet in self.controlnets:
            if hasattr(cnet, "cleanup"):
                cnet.cleanup()


# ------------------------------------------------------------
# 🎯 Main Node
# ------------------------------------------------------------
class JLC_ControlNetOrchestratorHyper:

    FUNCTION = "orchestrate"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
            "optional": {
                # --- SLOT 1 ---
                "image_01": ("IMAGE",),
                "strength_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_01": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "control_net_name_01": (
                    ["DISABLED"] + folder_paths.get_filename_list("controlnet"),
                    {
                        "tooltip": "ControlNet selector"
                    }
                ),

                # --- SLOT 2 ---
                "image_02": ("IMAGE",),
                "strength_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_02": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "control_net_name_02": (
                    ["DISABLED", "SHARE_PREVIOUS"] + folder_paths.get_filename_list("controlnet"),
                    {
                        "tooltip": "ControlNet selector"
                    }
                ),

                # --- SLOT 3 ---
                "image_03": ("IMAGE",),
                "strength_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_03": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "control_net_name_03": (
                    ["DISABLED", "SHARE_PREVIOUS"] + folder_paths.get_filename_list("controlnet"),
                    {
                        "tooltip": "ControlNet selector"
                    }
                ),

                # --- Order-dependent parameter. Breaks slot-invariance ---
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Order bias. 1.0 = neutral. <1 favors earlier slots. >1 favors later slots. Negative values invert influence."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")

    # ------------------------------------------------------------
    # 🧠 MAIN LOGIC
    # ------------------------------------------------------------
    def orchestrate(self, positive, negative, vae, **kwargs):

        # ------------------------------------------------------------
        # Phase 1 — Resolve slot pairing
        # ------------------------------------------------------------
        resolved = []

        alpha = kwargs.get("alpha", 1.0)
        
        # Checking only the incoming cnet
        current_base = None

        for i in range(1, 4):
            idx = f"{i:02d}"
            image = kwargs.get(f"image_{idx}")
            strength = kwargs.get(f"strength_{idx}", 1.0)
            start = kwargs.get(f"start_{idx}", 0.0)
            end = kwargs.get(f"end_{idx}", 1.0)
            weight = kwargs.get(f"weight_{idx}", 1.0)
            name = kwargs.get(f"control_net_name_{idx}")

            # ----------------------------------------
            # HARD BYPASS
            # ----------------------------------------
            if name == "DISABLED":
                continue

            # ----------------------------------------
            # SHARE PREVIOUS
            # ----------------------------------------
            if name == "SHARE_PREVIOUS":
                if current_base is None:
                    continue
                base = current_base

            # ----------------------------------------
            # LOAD / CACHE
            # ----------------------------------------
            else:
                path = folder_paths.get_full_path_or_raise("controlnet", name)

                if path not in GLOBAL_CONTROLNET_CACHE:
                    GLOBAL_CONTROLNET_CACHE[path] = comfy.controlnet.load_controlnet(path)

                base = GLOBAL_CONTROLNET_CACHE[path]

            current_base = base

            # ------------------------------------------------------------
            # 🚫 EARLY BYPASS (CRITICAL)
            # ------------------------------------------------------------
            if (
                image is None
                or strength == 0
                or weight == 0
                or (end - start) <= 0
            ):
                continue

            resolved.append({
                "slot": i,
                "base": base,
                "image": image,
                "strength": strength,
                "start": start,
                "end": end,
                "weight": weight,
            })

        active = [str(item["slot"]) for item in resolved]
        inactive = [str(i) for i in range(1, 4) if str(i) not in active]

        print(f"[JLC-Orchestrator] Active: {', '.join(active) or 'none'} | Inactive: {', '.join(inactive)}")

        if not resolved:
            return (positive, negative)

        # ------------------------------------------------------------
        # Phase 2 — Build independent ControlNets
        # ------------------------------------------------------------
        prepared_cnets = []
        weights = []

        for item in resolved:
            cnet_base = item["base"]
            image = item["image"]
            strength = item["strength"]
            weight = item["weight"]

            control_hint = image.movedim(-1, 1)

            cnet = (
                cnet_base
                .copy()
                .set_cond_hint(
                    control_hint,
                    strength,
                    (item["start"], item["end"]),
                    vae=vae,
                )
            )

            prepared_cnets.append(cnet)
            weights.append(weight)

        # ------------------------------------------------------------
        # 🟡 Special Case — Single ControlNet (bypass composition)
        # ------------------------------------------------------------
        if len(prepared_cnets) == 1:
            single_cnet = prepared_cnets[0]

            def inject_single(conditioning):
                out = []
                for t in conditioning:
                    d = t[1].copy()

                    prev_cnet = d.get('control', None)

                    # replicate Comfy's stock ApplyAdvanced behavior
                    c_net = single_cnet.copy()
                    c_net.set_previous_controlnet(prev_cnet)

                    d['control'] = c_net
                    d['control_apply_to_uncond'] = False

                    out.append([t[0], d])
                return out

            return (inject_single(positive), inject_single(negative))


        # ------------------------------------------------------------
        # Phase 3 — Compose
        # ------------------------------------------------------------
        final_weights = [
            w * (alpha ** i)
            for i, w in enumerate(weights)
        ]
        
        composed = JLC_ComposedControlNet(prepared_cnets, final_weights)

        # ------------------------------------------------------------
        # Phase 4 — Inject into conditioning
        # ------------------------------------------------------------
        def inject(conditioning):
            out = []
            for t in conditioning:
                d = t[1].copy()
                d["control"] = composed
                d["control_apply_to_uncond"] = False
                out.append([t[0], d])
            return out

        return (inject(positive), inject(negative))