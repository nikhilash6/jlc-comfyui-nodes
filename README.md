# JLC ComfyUI Nodes

<p align="center">
  <img src="assets/icons/jlc-comfyui-nodes_Logo-0512.png" width="120">
  &nbsp;&nbsp;&nbsp;
  <img src="assets/icons/jlc-comfyui-nodes_Logo-Dark-0512.png" width="120">
</p>

<p align="center">
  <a href="https://registry.comfy.org/packages/jlc-comfyui-nodes">
    <img src="https://img.shields.io/badge/Available%20on-ComfyUI%20Registry-blue" alt="ComfyUI Registry">
  </a>
  <img src="https://img.shields.io/badge/ComfyUI-Custom%20Nodes-blue" alt="ComfyUI Custom Nodes">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status: active">
</p>

> [!WARNING]
> **Temporary ControlNet compatibility notice**
>
> Recent ComfyUI sampler changes affected the older JLC ControlNet Composition and Orchestrator wrappers.
>
> We believe the issue is now fixed in the current `main` branch, including updated ControlNet Composition, Orchestrator, Advanced Orchestrator, and related frontend slot-visibility support. However, testing is still ongoing, so please proceed with caution.
>
> If you rely on the ControlNet Composition or Orchestrator nodes for production workflows, keep a backup of your working install before updating.
>
> Detailed documentation will follow shortly.
>
> Bug reports are welcome, especially with:
>
> - ComfyUI version or commit
> - JLC node used
> - Workflow screenshot or JSON, if shareable
> - Console log
> - GPU and VRAM amount


**JLC ComfyUI Nodes** is a practical custom-node collection for ComfyUI workflows that need cleaner ControlNet composition, padded inpainting/outpainting helpers, dynamic ControlNet auxiliary preprocessors, dynamic LoRA loading, and small workflow utilities.

The collection is developed by **J. L. Córdova** and is especially focused on Flux-oriented image generation pipelines, LoRA experimentation, ControlNet-heavy workflows, inpainting/outpainting, and multi-stage inference setups.

---

## Documentation

This README is the front page for the repository. Detailed documentation is split by node family:

1. [ControlNet Composition and Orchestration](docs/controlnet-composition.md)
2. [Padded Image / Padded Latent](docs/padded-image-latent.md)
3. [ControlNet Aux Preprocessor Wrappers](docs/controlnet-aux.md)
4. [Dynamic LoRA Loaders](docs/lora-loaders.md)
5. [Utility Nodes](docs/utility-nodes.md)

---

## Node Families

### 1. ControlNet Composition and Orchestrator

Nodes for replacing recursive ControlNet chains with explicit non-recursive weighted composition, plus supporting Apply nodes for native chained workflows.

This family includes:

- **JLC ControlNet Composition**
- **JLC ControlNet Orchestrator**
- **JLC ControlNet Orchestrator - Advanced Dynamic**
- **JLC ControlNet Apply**
- **JLC ControlNet Apply - Advanced**

The headline idea is simple: ComfyUI's native ControlNet behavior is recursive and chain-based; the JLC Composition and Orchestrator nodes provide an experimental parallel-fusion alternative for multi-ControlNet workflows that should result in significant inference speed in most configurations.

[Read the ControlNet guide](docs/controlnet-composition.md)

---

### 2. Padded Image / Padded Latent

Nodes for padded-canvas workflows, especially inpainting and outpainting.

This family includes:

- **JLC Padded Image**
- **JLC Inpaint-Conditioned Padded Latent**

Use **Padded Image** when you want image, mask, and canvas preparation while keeping VAE encoding and inpaint conditioning separate. Use **Padded Latent** when you want a more integrated node that prepares the padded canvas and also injects inpaint-conditioning metadata.

[Read the Padded Image / Padded Latent guide](docs/padded-image-latent.md)

---

### 3. ControlNet Aux Preprocessor Wrappers

A dynamic convenience wrapper for simple image-in/image-out preprocessors provided by **Fannovel16's `comfyui_controlnet_aux`** package.

This family currently includes:

- **JLC Dynamic Aux Preprocessor Wrapper**

The wrapper does not replace Fannovel16's nodes and does not claim ownership of the underlying preprocessors. It provides a compact multi-slot interface for JLC workflows that need several simple ControlNet hint images from the same source image. Parameter-heavy preprocessors should still be used through their native ControlNet Aux nodes.

`comfyui_controlnet_aux` must be installed for non-disabled preprocessor slots to run.

[Read the ControlNet Aux guide](docs/controlnet-aux.md)

---

### 4. Dynamic LoRA Loaders

Dynamic LoRA loader nodes for MODEL-only and MODEL+CLIP workflows, including shared and per-slot block-weight variants.

This family includes:

- **JLC LoRA Loader - Multi Model**
- **JLC LoRA Loader - Multi-Model / CLIP**
- **JLC LoRA Loader - Multi-Model / Shared Block Weight**
- **JLC LoRA Loader - Multi Model / Shared Block Weight + CLIP**
- **JLC LoRA Loader - Multi-Model / Block Weight**
- **JLC LoRA Loader - Multi-Model / CLIP + Block Weight**

These nodes predeclare up to ten LoRA slots and use frontend visibility controls to expose only the active rows. Hidden slot values remain serialized in workflow JSON, while the backend treats `slot_count` as authoritative. MODEL-only variants intentionally avoid CLIP/text-encoder patching; MODEL+CLIP variants expose independent MODEL and CLIP strengths.

[Read the LoRA Loader guide](docs/lora-loaders.md)

---

### 5. Utility Nodes

Small workflow support nodes for seed discipline and stage-boundary memory hygiene.

This family includes:

- **JLC Seed Generator** — shared seed source that keeps the visible base seed stable while a frontend display reports the last seed actually used.
- **JLC Stage Boundary VRAM Cleanup** — experimental latent-passthrough cleanup helper for advanced multi-stage workflows where selected heavy model objects should be unloaded before the next stage.

[Read the Utility Nodes guide](docs/utility-nodes.md)

---

The repository includes example ComfyUI workflows in `assets/workflows/`. PNG workflows contain embedded ComfyUI graphs and can be dragged directly onto the ComfyUI canvas.

The front-page showcase workflow for this release is a compact ControlNet orchestration example. It is intended to demonstrate a practical combination of JLC nodes in one workflow, including dynamic LoRA loading, ControlNet Aux preprocessing, seed/display utility behavior, and JLC ControlNet orchestration.

## Release 1.5 Showcase Workflow

![JLC Orchestrator Showcase Workflow](assets/workflows/Release_1.5/Orchestrator_Workflow.png)

[Download PNG workflow](assets/workflows/Release_1.5/Orchestrator_Workflow.png) ·
[Download JSON workflow](assets/workflows/Release_1.5/Orchestrator_Workflow.json)

PNG workflows contain embedded ComfyUI graphs and may be dragged directly into the ComfyUI canvas. The JSON version is provided as a plain workflow backup for users who prefer explicit import files.

---

## Installation

Install through the **ComfyUI Registry**:

[https://registry.comfy.org/packages/jlc-comfyui-nodes](https://registry.comfy.org/packages/jlc-comfyui-nodes)

Manual install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Damkohler/jlc-comfyui-nodes.git
```

Restart ComfyUI after installation or update.

To update manually:

```bash
cd ComfyUI/custom_nodes/jlc-comfyui-nodes
git pull
```

### Optional dependency: ControlNet Aux preprocessors

The **JLC Dynamic Aux Preprocessor Wrapper** requires Fannovel16's `comfyui_controlnet_aux` package when any non-disabled preprocessor slot is used.

Install it through ComfyUI Manager, or follow the upstream installation instructions:

[https://github.com/Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)

Some upstream preprocessors may download or load large auxiliary models the first time they are used.

---

## Compatibility Notes

The nodes are designed for ComfyUI custom-node workflows and have been developed primarily around:

- Flux-based image generation workflows
- ControlNet-heavy pipelines
- LoRA experimentation
- inpainting and outpainting workflows
- multi-stage inference graphs

Some ControlNet composition/orchestration nodes intentionally experiment with non-canonical ControlNet execution. They are documented as experimental where appropriate.

---

## Repository Structure

The current repository includes these main areas:

```text
nodes/
  controlnet_aux_nodes/
  engines/
  lora_loader_nodes/
  util_nodes/
web/
assets/
  icons/
  workflows/
docs/
```

A generated repository map is also included:

```text
jlc-comfyui-nodes-structure.txt
```

---

## License

MIT License.

---

## Author

**J. L. Córdova**  
GitHub: [Damkohler](https://github.com/Damkohler)

---

## Contributions

Suggestions, bug reports, and improvements are welcome through GitHub issues or pull requests.
