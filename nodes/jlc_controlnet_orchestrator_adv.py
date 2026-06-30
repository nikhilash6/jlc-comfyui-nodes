"""
JLC ControlNet Orchestrator Advanced
------------------------------------

Maintenance-rescue version for internal ControlNet loading.

This node keeps the April-style non-recursive orchestration algorithm intact.
The shared JLC model cache is used only for resident base ControlNet objects
loaded by dropdown name.  Per-slot execution still uses independent `.copy()`
isolation exactly as before.

No `copy.deepcopy()` and no real MultiGPU deep-clone support are introduced.
"""

MANIFEST = {
    "name": "JLC ControlNet Orchestrator Advanced",
    "version": (1, 1, 1),
    "author": "J. L. Córdova",
    "description": (
        "Internal-loader ControlNet orchestrator with shared JLC cache, "
        "non-recursive weighted fusion, compatibility shims, and dynamic slots."
    ),
}

import torch
import folder_paths
import comfy.controlnet

try:
    from .engines.jlc_model_cache_core import (
        get_controlnet_cache_capacity,
        get_or_load_model,
        make_controlnet_cache_key,
    )
except Exception:
    from jlc_model_cache_core import (  # type: ignore
        get_controlnet_cache_capacity,
        get_or_load_model,
        make_controlnet_cache_key,
    )

MAX_SLOTS = 10
DEBUG = True
DISABLED = "DISABLED"
SHARE_PREVIOUS = "SHARE_PREVIOUS"


def _default_cache_size():
    try:
        return max(0, int(get_controlnet_cache_capacity()))
    except Exception:
        return 2


def _clear_multigpu_clone_state(controlnet):
    if hasattr(controlnet, "multigpu_clones"):
        controlnet.multigpu_clones = {}
    return controlnet


def _safe_cnet_name(controlnet):
    model = getattr(controlnet, "control_model", None)
    if model is not None:
        return model.__class__.__name__
    return controlnet.__class__.__name__


def _load_controlnet_with_shared_cache(control_net_name, cache_size=None):
    controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
    key = make_controlnet_cache_key(controlnet_path)

    def loader():
        print(f"[JLC-ControlNet Cache] Loading ControlNet: {control_net_name}")
        cnet = comfy.controlnet.load_controlnet(controlnet_path)
        if cnet is None:
            raise RuntimeError(f"Invalid ControlNet model file: {control_net_name}")
        return cnet

    max_loaded_for_family = None
    if cache_size is not None:
        max_loaded_for_family = max(0, int(cache_size))

    return get_or_load_model(
        key,
        loader,
        family="controlnet",
        model_path=controlnet_path,
        role="controlnet",
        policy="lru_family_capacity",
        max_loaded_for_family=max_loaded_for_family,
        metadata={"control_net_name": control_net_name},
    )


class JLC_ComposedControlNet:
    def __init__(self, controlnets, weights):
        self.controlnets = controlnets
        self.weights = weights
        self.previous_controlnet = None
        self.extra_hooks = None
        self.multigpu_clones = {}

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        combined = None

        for cnet, w in zip(self.controlnets, self.weights):
            if cnet is None or w == 0:
                continue

            out = cnet.get_control(x_noisy, t, cond, batched_number, transformer_options)
            if out is None:
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if combined is None:
                combined = {}
                for key, out_list in out.items():
                    new_list = [None] * len(out_list)
                    for i, v in enumerate(out_list):
                        if v is None:
                            continue
                        owned = v.clone()
                        if w != 1.0:
                            owned.mul_(w)
                        new_list[i] = owned
                    combined[key] = new_list
            else:
                for key, out_list in out.items():
                    if key not in combined:
                        combined[key] = [None] * len(out_list)
                    combined_list = combined[key]
                    if len(combined_list) < len(out_list):
                        combined_list.extend([None] * (len(out_list) - len(combined_list)))
                    for i, v in enumerate(out_list):
                        if v is None:
                            continue
                        dst = combined_list[i]
                        if dst is None:
                            owned = v.clone()
                            if w != 1.0:
                                owned.mul_(w)
                            combined_list[i] = owned
                        else:
                            dst.add_(v, alpha=w)
                    del out_list

            del out

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return combined

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

    def get_models_only_self(self):
        return self.get_models()

    def get_instance_for_device(self, device):
        return self

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


class JLC_ControlNetOrchestratorAdvanced:
    FUNCTION = "orchestrate"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        controlnet_names = folder_paths.get_filename_list("controlnet")

        optional = {
            "slot_count": ("INT", {
                "default": 3,
                "min": 1,
                "max": MAX_SLOTS,
                "step": 1,
                "tooltip": "Number of visible/active internal ControlNet slots. Backend ignores slots above this count.",
            }),
            "controlnet_cache_size": ("INT", {
                "default": _default_cache_size(),
                "min": 0,
                "max": 10,
                "step": 1,
                "advanced": True,
                "tooltip": "Shared JLC ControlNet cache capacity. 0 means evict/prevent resident cached ControlNets.",
            }),
        }

        for i in range(1, MAX_SLOTS + 1):
            idx = f"{i:02d}"
            choices = [DISABLED] + ([] if i == 1 else [SHARE_PREVIOUS]) + controlnet_names
            optional[f"control_net_name_{idx}"] = (choices, {
                "tooltip": "ControlNet model for this slot. SHARE_PREVIOUS reuses the last selected model."
            })
            optional[f"image_{idx}"] = ("IMAGE", {
                "tooltip": "Control image for this slot."
            })
            optional[f"strength_{idx}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            optional[f"start_{idx}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001})
            optional[f"end_{idx}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            optional[f"weight_{idx}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        optional["alpha"] = ("FLOAT", {
            "default": 1.0,
            "min": -2.0,
            "max": 2.0,
            "step": 0.01,
            "tooltip": "Order bias. 1.0 = neutral. <1 favors earlier slots. >1 favors later slots. Negative values invert influence.",
        })

        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")

    def orchestrate(self, positive, negative, vae, slot_count=3, controlnet_cache_size=None, alpha=1.0, **kwargs):
        slot_count = max(1, min(MAX_SLOTS, int(slot_count)))
        cache_size = _default_cache_size() if controlnet_cache_size is None else int(controlnet_cache_size)

        resolved = []
        current_base = None
        current_name = None

        for i in range(1, slot_count + 1):
            idx = f"{i:02d}"
            name = kwargs.get(f"control_net_name_{idx}", DISABLED)
            image = kwargs.get(f"image_{idx}")
            strength = kwargs.get(f"strength_{idx}", 1.0)
            start = kwargs.get(f"start_{idx}", 0.0)
            end = kwargs.get(f"end_{idx}", 1.0)
            weight = kwargs.get(f"weight_{idx}", 1.0)

            if name in (None, "", DISABLED):
                continue

            if name == SHARE_PREVIOUS:
                if current_base is None:
                    continue
                base = current_base
                resolved_name = current_name or SHARE_PREVIOUS
            else:
                base = _load_controlnet_with_shared_cache(name, cache_size=cache_size)
                current_base = base
                current_name = name
                resolved_name = name

            if (
                image is None
                or strength == 0
                or weight == 0
                or (end - start) <= 0
            ):
                continue

            resolved.append({
                "slot": i,
                "name": resolved_name,
                "base": base,
                "image": image,
                "strength": strength,
                "start": start,
                "end": end,
                "weight": weight,
            })

        if DEBUG:
            active = [str(item["slot"]) for item in resolved]
            inactive = [str(i) for i in range(1, slot_count + 1) if str(i) not in active]
            print(f"[JLC-Orchestrator-Advanced] Active: {', '.join(active) or 'none'} | Inactive: {', '.join(inactive) or 'none'}")

        if not resolved:
            return (positive, negative)

        prepared_cnets = []
        weights = []

        for item in resolved:
            control_hint = item["image"].movedim(-1, 1)
            cnet = (
                item["base"]
                .copy()
                .set_cond_hint(
                    control_hint,
                    item["strength"],
                    (item["start"], item["end"]),
                    vae=vae,
                )
            )
            _clear_multigpu_clone_state(cnet)
            prepared_cnets.append(cnet)
            weights.append(item["weight"])

        if len(prepared_cnets) == 1:
            return self._inject_single_native(positive, negative, prepared_cnets[0])

        final_weights = [w * (alpha ** i) for i, w in enumerate(weights)]

        if DEBUG:
            names = [item["name"] for item in resolved]
            print(f"[JLC-Orchestrator-Advanced] Composed slots={names} weights={final_weights} alpha={alpha}")

        composed = JLC_ComposedControlNet(prepared_cnets, final_weights)
        return (self._inject_composed(positive, composed), self._inject_composed(negative, composed))

    def _inject_single_native(self, positive, negative, single_cnet):
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
                    c_net = single_cnet.copy()
                    _clear_multigpu_clone_state(c_net)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                c.append([t[0], d])
            out.append(c)

        return (out[0], out[1])

    @staticmethod
    def _inject_composed(conditioning, composed):
        out = []
        for t in conditioning:
            d = t[1].copy()
            d["control"] = composed
            d["control_apply_to_uncond"] = False
            out.append([t[0], d])
        return out
    

NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetOrchestratorAdvanced": JLC_ControlNetOrchestratorAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetOrchestratorAdvanced": "JLC ControlNet Orchestrator (Advanced)",
}
