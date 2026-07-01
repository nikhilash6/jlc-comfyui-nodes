# ControlNet Aux Preprocessor Wrappers

This chapter covers the JLC ControlNet Aux wrapper family:

- [JLC Dynamic Aux Preprocessor Wrapper](#jlc-dynamic-aux-preprocessor-wrapper)
- [Dependency on Fannovel16's ControlNet Aux package](#dependency-on-fannovel16s-controlnet-aux-package)
- [Why the wrapper is intentionally limited](#why-the-wrapper-is-intentionally-limited)
- [Choosing native Aux nodes vs. the JLC wrapper](#choosing-native-aux-nodes-vs-the-jlc-wrapper)
- [First-use model downloads](#first-use-model-downloads)
- [Example Workflows](#example-workflows)

The JLC wrapper is a convenience layer for specific JLC workflows. It depends on the upstream ControlNet Auxiliary Preprocessors package and does not replace the original nodes.

---

## Dependency on Fannovel16's ControlNet Aux package

The **JLC Dynamic Aux Preprocessor Wrapper** sits on top of **Fannovel16's `comfyui_controlnet_aux`** package.

The upstream project provides the actual ControlNet auxiliary preprocessors. The JLC node only provides a compact multi-slot wrapper around a carefully limited subset of those preprocessors.

Install the upstream dependency through ComfyUI Manager or from the upstream repository:

[https://github.com/Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)

If the package is not installed, disabled slots can remain idle, but any active non-disabled preprocessor slot will raise an error explaining that `comfyui_controlnet_aux` is required.

### Attribution

The preprocessor implementations come from the upstream ControlNet Aux ecosystem. This JLC node is not claiming originality for those preprocessors and is not intended to supplant the upstream package.

Use the upstream/native nodes whenever you need the full capability of a specific preprocessor.

---

## JLC Dynamic Aux Preprocessor Wrapper

**JLC Dynamic Aux Preprocessor Wrapper** is a compact multi-slot wrapper for simple image-in/image-out preprocessors.

It is designed for workflows where one source image is sent through several simple preprocessors to create multiple ControlNet hint images.

Example intent:

```text
source image
    ↓
JLC Dynamic Aux Preprocessor Wrapper
    ├─ image_01: HED-style hint
    ├─ image_02: depth-style hint
    ├─ image_03: normal-style hint
    └─ ...
```

Those output images can then be connected to ControlNet Apply, ControlNet Composition, or ControlNet Orchestrator workflows.

### Dynamic slot behavior

The node predeclares ten output slots and ten preprocessor selector widgets.

`slot_count` is authoritative:

- slots `1..slot_count` are visible/executed;
- slots above `slot_count` may remain serialized in workflow JSON;
- hidden slots are ignored by backend execution;
- hidden output sockets are managed by frontend JavaScript;
- static output arity remains stable for ComfyUI compatibility.

The node returns ten `IMAGE` outputs:

```text
image_01
image_02
...
image_10
```

For hidden slots, the backend returns the original image as a passthrough placeholder so the static return shape remains stable.

### Shared resolution

The wrapper exposes one shared `resolution` input.

Only preprocessors that fit the simple shared-resolution model are exposed:

```text
IMAGE input
optional/shared resolution input
IMAGE-only output
```

This is what keeps the wrapper compact and predictable.

---

## Why the wrapper is intentionally limited

The wrapper intentionally exposes only simple preprocessors.

Accepted shape:

```text
required input: image
optional input: resolution
return type: IMAGE only
```

Excluded preprocessors include nodes that require or produce:

- thresholds;
- detector toggles;
- model selectors;
- pose/keypoint JSON;
- masks;
- optical-flow payloads;
- segmentation payloads;
- custom output types;
- other special parameters.

This is deliberate. A generic compact wrapper should not hide important preprocessor-specific controls.

For example, parameter-heavy or special-output preprocessors such as Canny, OpenPose, DWPose, segmentation, optical flow, and SAM-style nodes should be used through their native ControlNet Aux nodes instead.

---

## Curated list and autodiscovery

The JLC wrapper uses a curated include/exclude policy:

- a blacklist excludes known non-simple or special-case families;
- a preferred whitelist controls ordering for common simple preprocessors;
- autodiscovery may include currently installed aux preprocessors that pass the strict structural gate.

This keeps the wrapper useful as the upstream package evolves, while still preventing parameter-heavy nodes from entering the compact UI.

The exact set of available options depends on the installed version of `comfyui_controlnet_aux`.

---

## Fallback behavior when the dependency is missing

The wrapper provides a fallback dropdown list even if `comfyui_controlnet_aux` cannot be imported.

This is intentional. It helps avoid confusing ComfyUI workflow-validation errors when opening a saved workflow that references a preprocessor name.

However, execution still requires the upstream package:

```text
DISABLED slots       → safe no-op
active aux slot      → requires comfyui_controlnet_aux
```

If an active selected preprocessor cannot run, the node raises a clear runtime error.

---

## First-use model downloads

Some ControlNet Aux preprocessors rely on auxiliary models.

Depending on the upstream preprocessor and local cache state, the first use of a selected preprocessor may:

- download a model;
- load a large model into memory;
- take noticeably longer than later runs;
- require internet access if the model is not already cached;
- require enough local disk space for the downloaded files.

This behavior belongs to the upstream preprocessor package and model ecosystem, not to the JLC wrapper itself.

Plan for a slower first run when testing a new preprocessor.

---

## Choosing native Aux nodes vs. the JLC wrapper

| Need | Recommended choice |
|---|---|
| One compact node that produces several simple hint images from one source image | JLC Dynamic Aux Preprocessor Wrapper |
| Shared resolution across several simple preprocessors | JLC Dynamic Aux Preprocessor Wrapper |
| Thresholds, detector toggles, or model-specific controls | Native `comfyui_controlnet_aux` node |
| Pose/keypoint outputs or JSON output | Native `comfyui_controlnet_aux` node |
| Masks, segmentation, optical flow, SAM-style processing, or special payloads | Native `comfyui_controlnet_aux` node |
| Maximum fidelity to the upstream node UI | Native `comfyui_controlnet_aux` node |

A simple rule:

```text
Use the JLC wrapper for simple IMAGE → IMAGE preprocessors.
Use native Aux nodes when the preprocessor has meaningful controls or special outputs.
```

---

## Example Workflows

Detailed ControlNet Aux workflow examples will be added shortly.

Planned examples:

- one input image processed into multiple hint images;
- Aux wrapper outputs feeding **JLC ControlNet Orchestrator - Advanced Dynamic**;
- Aux wrapper outputs feeding **JLC ControlNet Composition** after native Apply nodes;
- comparison workflow showing when to use native Aux nodes instead of the wrapper.

Workflow will be added shortly.

---

## Notes for Advanced Users

### Validation gate

The wrapper inspects each upstream aux node before exposing it. A node must have an `INPUT_TYPES` method, a callable function, compatible inputs, and exactly one `IMAGE` return.

### Autodiscovery

Autodiscovery is enabled for simple-compatible preprocessors so the local JLC list does not become stale as the upstream package grows.

### Saved workflow values

Hidden slot values remain serialized in workflow JSON. This allows users to reduce `slot_count` without losing the selected preprocessor values in hidden rows.

### Why hidden outputs passthrough the source image

ComfyUI expects the node's declared return shape to stay fixed. Returning the source image for ignored hidden slots keeps that return shape stable while making `slot_count` authoritative for actual preprocessing.
