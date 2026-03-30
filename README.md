# JLC ComfyUI Nodes

<p align="center">
  <img src="assets/icons/jlc-comfyui-nodes_Logo-0512.png" width="120" style="vertical-align: middle;">
  &nbsp;&nbsp;&nbsp;
  <img src="assets/icons/jlc-comfyui-nodes_Logo-Dark-0512.png" width="120" style="vertical-align: middle;">
</p>

[![ComfyUI Registry](https://img.shields.io/badge/Available%20on-ComfyUI%20Registry-blue)](https://registry.comfy.org/packages/jlc-comfyui-nodes)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## Featured Node in this Release: JLC ControlNet Apply (Advanced)

The **JLC ControlNet Apply (Advanced)** node is an extension of ComfyUI’s native ControlNet application logic.

It preserves full compatibility with ComfyUI’s conditioning pipeline while introducing:

- integrated ControlNet loading  
- bounded session-level caching (LRU)  
- deterministic reuse across nodes  
- mutation-safe execution  

The node is designed to improve workflow efficiency **without altering ComfyUI’s execution semantics**.

---

### Architecture Overview

#### Stateless Execution

ControlNet application follows ComfyUI’s native pattern:

- `control_net.copy().set_cond_hint(...)`
- `set_previous_controlnet(...)`

The original ControlNet object is never mutated during execution.

---

#### Three-Tier Model Resolution

ControlNet objects are resolved in strict priority:

1. Wired input (`control_net`)
2. Internal LRU cache
3. Disk load (`control_net_name`)

This ensures deterministic behavior while enabling reuse when possible.

---

#### Bounded LRU Cache

- Implemented via `OrderedDict`
- Configurable size (default: 2–3 models)
- Session-scoped (cleared on restart)

Behavior:
- Reuse → `move_to_end()`
- Overflow → `popitem(last=False)`

Cache reduces redundant disk loads while remaining fully controlled.

---

#### Mutation Safety

Cached ControlNet objects are automatically cleaned before reuse.

This guarantees:
- no cross-node contamination
- no cross-run state leakage
- safe reuse without deep copies

---

#### Determinism

- Wired reuse → fully deterministic  
- Cached reuse → deterministic via cleanup  
- No weak references  
- No GC-dependent behavior  

All reuse paths are explicit and controlled.

---

#### Performance Scope

Improves:
- ControlNet load efficiency  
- multi-node reuse within a session  

Does not affect:
- VRAM usage during sampling  
- ControlNet compute cost  

VRAM remains managed entirely by ComfyUI.

---

### Design Principle

> Safe reuse requires preserving a clean base object, not avoiding reuse entirely.

---

A collection of workflow-focused ComfyUI nodes designed to simplify advanced image generation pipelines. Includes tools for flexible image padding and mask merging to enable inpainting and outpainting in a single pass, sequential LoRA stacking (up to 10 LoRAs), a two LoRA loader with block-weight control, and reusable components for Flux-based workflows and complex image generation pipelines.

Developed by **J. L. Córdova**.

These nodes are designed for:

- Flux workflows  
- LoRA experimentation  
- advanced inpainting / outpainting pipelines  
- reusable pipeline components  

Repository  
https://github.com/Damkohler/jlc-comfyui-nodes

---

## Example Workflows

PNG workflows contain the embedded ComfyUI graph and can be dragged directly into the ComfyUI canvas.

### ControlNet Workflow

<p align="center">
  <img src="assets/workflows/jlc_ControlNet_Apply_Advanced.png" width="900">
</p>

<p align="center">
  <a href="assets/workflows/jlc_ControlNet_Apply_Advanced.png">Download PNG</a> •
  <a href="assets/workflows/jlc_ControlNet_Apply_Advanced.json">Download JSON</a>
</p>


---

### Basic Inpainting / Outpainting Workflow Using JLC Padded Image

<p align="center">
  <img src="assets/workflows/jlc_padded_image_Basic_Infill_Outfill.png" width="900">
</p>

<p align="center">
  <a href="assets/workflows/jlc_padded_image_Basic_Infill_Outfill.png">Download PNG</a> •
  <a href="assets/workflows/jlc_padded_image_Basic_Infill_Outfill.json">Download JSON</a>
</p>

---

### Preferred Inpainting / Outpainting Workflow Using JLC Padded Image

<p align="center">
  <img src="assets/workflows/jlc_padded_image_Best_Infill_Outfill.png" width="900">
</p>

<p align="center">
  <a href="assets/workflows/jlc_padded_image_Best_Infill_Outfill.png">Download PNG</a> •
  <a href="assets/workflows/jlc_padded_image_Best_Infill_Outfill.json">Download JSON</a>
</p>

*(Other workflows will be added later)*

---

## Table of Contents

- [Installation](#installation)
- [Nodes Included](#nodes-included)
- [Node Descriptions](#node-descriptions)
- [Design Philosophy](#design-philosophy)
- [Compatibility](#compatibility)
- [License](#license)
- [Attribution](#attribution)
- [Author](#author)
- [Future Plans](#future-plans)
- [Contributions](#contributions)

---

## Installation

### Using git:

Clone this repository into your **ComfyUI `custom_nodes` directory**.

```
ComfyUI/
└── custom_nodes/
```

Then run:

```
git clone https://github.com/Damkohler/jlc-comfyui-nodes.git
```


Restart **ComfyUI** after installation.

---

### Install via ComfyUI Manager

1. Open **ComfyUI**
2. Open **Manager**
3. Search for **JLC ComfyUI Nodes**
4. Click **Install**

---

# Nodes Included

| Node | Purpose |
|-----|--------|
| **JLC Padded Image** | Canvas preparation for inpainting and outpainting workflows |
| **JLC Padded Latent** | Combined padded-image + latent + mask conditioning pipeline |
| **JLC ControlNet Apply** | Legacy ControlNet node (simplified application) |
| **JLC ControlNet Apply (Advanced)** | Advanced ControlNet application with caching and deterministic reuse |
| **JLC 10 LoRA Loader Stack** | Sequential loader for up to 10 LoRAs |
| **JLC LoRA Loader (Block Weight)** | Multi-slot LoRA loader with block weight control |

---

# Node Descriptions

## JLC Padded Image

A utility node that prepares images for **inpainting or outpainting** by placing them on a new canvas with a specified aspect ratio and size.

### Features

- Canvas resizing with aspect ratio control  
- Image placement using offset controls  
- High-quality **Lanczos resampling**  
- Automatic outpaint mask generation  
- Optional manual mask merging  
- Deterministic padding behavior  

Designed to work particularly well with inpainting models such as `flux1-fill-dev`.

---

## JLC Padded Latent

A higher-level workflow node that combines:

- padded image preparation  
- outpaint mask generation and merge with inpaint masks  
- inpaint conditioning  

### Outputs

- conditioned positive prompt  
- conditioned negative prompt  
- latent image  
- mask  
- image dimensions  

This node simplifies building **reusable inpainting pipelines**.

---

## JLC ControlNet Apply

A streamlined node for applying **ControlNet conditioning** within a generation pipeline.

### Design Goals

- simplified parameter handling  
- improved workflow clarity  
- compatibility with Flux-based pipelines  

This node adapts the built-in **ComfyUI ControlNet application logic** for cleaner integration into custom workflows.

---

## JLC ControlNet Apply (Advanced)

An advanced ControlNet application node that combines:

- model loading  
- deterministic chaining  
- session-level caching  

### Key Features

- supports both wired and internal ControlNet sources  
- avoids redundant model loads via LRU cache  
- preserves ComfyUI conditioning behavior  
- mutation-safe reuse across nodes  

Designed for complex workflows requiring multiple ControlNet applications reducing unnecessary overhead.

---

## JLC 10 LoRA Loader Stack

Applies up to **ten LoRA models sequentially** to a base model.

### Features

Each slot includes:

- selectable LoRA file  
- independent strength control  

Slots operate independently and are applied **in order**.

Empty slots or strengths of zero are automatically skipped.

### Inspiration

Concept inspired by:

https://github.com/rgthree

---

## JLC LoRA Loader (Block Weight)

A LoRA loader with **block weight support**, allowing detailed control over how LoRA influence is distributed across model layers.

### Features

- multiple LoRA slots  
- independent model and CLIP strengths  
- per-slot block weight vectors  
- sequential LoRA application  

Adapted from:

https://github.com/ltdrdata/ComfyUI-Inspire-Pack

---

# Design Philosophy

- Workflow clarity  
- Deterministic behavior  
- Reusable building blocks  
- Clean integration with ComfyUI pipelines  

---

# Compatibility

Tested with:

- **ComfyUI**
- **Flux-based models**
- **LoRA-enabled pipelines**

---

# License

MIT License

---

# Author

**J. L. Córdova**  
https://github.com/Damkohler

---

# Future Plans

- Expand pipeline utilities  
- Improve instrumentation and debugging visibility  
- Continue building workflow-focused nodes  

---

# Contributions

Suggestions and improvements are welcome.  
Feel free to open issues or submit pull requests.

---