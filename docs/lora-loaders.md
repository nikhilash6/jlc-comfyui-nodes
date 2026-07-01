# Dynamic LoRA Loaders

This chapter covers the JLC Dynamic LoRA Loader family:

- [Core Dynamic Slot Design](#core-dynamic-slot-design)
- [MODEL-only vs. MODEL+CLIP Loaders](#model-only-vs-modelclip-loaders)
- [Plain Dynamic LoRA Loaders](#plain-dynamic-lora-loaders)
- [Shared Block-Weight Loaders](#shared-block-weight-loaders)
- [Per-Slot Block-Weight Loaders](#per-slot-block-weight-loaders)
- [Block Vector Notes](#block-vector-notes)
- [Legacy Compatibility Wrappers](#legacy-compatibility-wrappers)
- [Choosing the Right LoRA Loader](#choosing-the-right-lora-loader)
- [Example Workflows](#example-workflows)

These nodes replace fixed-count LoRA stack variants with dynamic loader nodes that expose only the active slot rows while preserving hidden slot values in the workflow.

---

## Core Dynamic Slot Design

The dynamic LoRA loaders all follow the same basic design:

```text
Python backend:
    predeclares up to 10 LoRA slots

Frontend JavaScript:
    hides or shows slot rows based on slot_count

Backend execution:
    treats slot_count as authoritative
```

This means:

- all ten slot widget groups can exist in the workflow payload;
- `slot_count` controls which slots are active;
- slots above `slot_count` are ignored by the backend;
- hidden slot values remain serialized in workflow JSON;
- reducing `slot_count` does not destroy hidden slot selections or strengths;
- increasing `slot_count` later can restore those hidden values;
- active LoRAs are applied sequentially in visible slot order.

The slot-count approach is intentionally conservative. It avoids runtime graph mutation while still making large LoRA stacks much easier to manage on the canvas.

---

## MODEL-only vs. MODEL+CLIP Loaders

The LoRA loader family is split into two broad groups.

### MODEL-only loaders

MODEL-only loaders return only a `MODEL`.

They patch the diffusion model / UNet side and intentionally do **not** patch CLIP or text encoders.

When a LoRA file contains text-encoder or CLIP-side keys, MODEL-only loaders filter those keys before MODEL-only application. This avoids misleading text-encoder warnings while keeping MODEL-side behavior explicit.

Use MODEL-only loaders when:

- your workflow does not need LoRA text-encoder effects;
- you are using a model family where text-encoder LoRA patching is not desired;
- you want the LoRA stack to affect the generated image model only;
- you want simpler wiring with a `MODEL` output only.

### MODEL+CLIP loaders

MODEL+CLIP loaders return both `MODEL` and `CLIP`.

Each active LoRA slot exposes independent strengths:

```text
strength_model
strength_clip
```

Use MODEL+CLIP loaders when:

- the LoRA was trained with useful text-encoder / CLIP-side behavior;
- the workflow expects CLIP to be patched;
- you want separate control over diffusion-model and text-encoder influence.

---

## Plain Dynamic LoRA Loaders

### JLC LoRA Loader - Multi Model

**JLC LoRA Loader - Multi Model** is the plain MODEL-only dynamic loader.

Each active slot provides:

- LoRA selector;
- MODEL strength.

The node returns:

```text
MODEL
```

CLIP is never patched. Text-encoder / CLIP LoRA keys are ignored by design before loading.

Use this as the default choice for dynamic MODEL-only LoRA stacks.

---

### JLC LoRA Loader - Multi-Model / CLIP

**JLC LoRA Loader - Multi-Model / CLIP** is the plain MODEL+CLIP dynamic loader.

Each active slot provides:

- LoRA selector;
- MODEL strength;
- CLIP/text-encoder strength.

The node returns:

```text
MODEL
CLIP
```

Use this as the default choice when a workflow should patch both the diffusion model and CLIP/text encoder.

---

## Shared Block-Weight Loaders

Block-weight loaders let LoRA influence vary across MODEL/UNet blocks.

A **shared block-weight** loader uses one `block_vector` for all active LoRA slots.

### JLC LoRA Loader - Multi-Model / Shared Block Weight

This is the MODEL-only shared-block-weight loader.

Each active slot provides:

- LoRA selector;
- MODEL strength.

One shared `block_vector` controls MODEL/UNet block weighting for every active LoRA.

The node returns:

```text
MODEL
```

CLIP is not patched.

Use this when you want a consistent block-weight profile across a stack of MODEL-only LoRAs.

---

### JLC LoRA Loader - Multi Model / Shared Block Weight + CLIP

This is the MODEL+CLIP shared-block-weight loader.

Each active slot provides:

- LoRA selector;
- MODEL strength;
- CLIP/text-encoder strength.

One shared `block_vector` controls MODEL/UNet block weighting for all active LoRAs. CLIP/text-encoder patches use ordinary per-slot CLIP strength and are not block-weighted.

The node returns:

```text
MODEL
CLIP
```

Use this when you want shared MODEL block weighting but still want ordinary CLIP patching.

---

## Per-Slot Block-Weight Loaders

Per-slot block-weight loaders expose a separate `block_vector` for each active LoRA slot.

This is more flexible, but also more visually dense.

### JLC LoRA Loader - Multi-Model / Block Weight

This is the MODEL-only per-slot block-weight loader.

Each active slot provides:

- LoRA selector;
- MODEL strength;
- per-slot MODEL block vector.

The node returns:

```text
MODEL
```

CLIP is not patched.

Use this when each MODEL-only LoRA needs its own block-weight profile.

---

### JLC LoRA Loader - Multi-Model / CLIP + Block Weight

This is the MODEL+CLIP per-slot block-weight loader.

Each active slot provides:

- LoRA selector;
- MODEL strength;
- CLIP/text-encoder strength;
- per-slot MODEL block vector.

The node returns:

```text
MODEL
CLIP
```

MODEL patches use the per-slot block vector. CLIP/text-encoder patches use ordinary per-slot CLIP strength.

Use this when each LoRA needs independent MODEL block weighting while still allowing CLIP/text-encoder influence.

---

## Block Vector Notes

Block vectors are numeric CSV strings.

Example:

```text
1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1
```

The default vector is:

```text
1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1
```

Conceptually:

- `vector[0]` is the base ratio for unmatched or "other" MODEL keys;
- `vector[1:]` is consumed across encountered MODEL block indices;
- if the vector runs out, the last value is reused;
- zero values suppress matching patches;
- nonzero values scale the slot's MODEL strength for matching patches.

The helper recognizes common block families such as:

```text
input_blocks
middle_block
output_blocks
double_blocks
single_blocks
```

Block vectors apply only to MODEL/UNet patches. CLIP/text-encoder patches are not block-weighted.

### Practical advice

Start with the plain loaders unless you know why you need block weighting.

Use shared block weighting when several LoRAs should receive the same MODEL block profile.

Use per-slot block weighting when different LoRAs need different MODEL block profiles.

---

## LoRA State Caching

The dynamic LoRA core uses a per-node-instance LoRA state cache.

When a LoRA file is selected and loaded, its state dictionary can be reused by that node instance instead of rereading the same LoRA file repeatedly during the same session.

This is a lightweight loader convenience. It does not change the sequential application order or the visible slot semantics.

---

## Legacy Compatibility Wrappers

The repository includes minimal legacy compatibility wrappers for selected old JLC LoRA loader class types.

These wrappers are provided to reduce breakage in older saved workflows. New workflows should use the dynamic LoRA loader nodes directly.

Retained legacy compatibility targets include:

- fixed 10-slot LoRA stack;
- fixed 2-slot block-weight LoRA stack.

The old family of ad-hoc fixed-count stack variants is intentionally not preserved as a primary interface.

Legacy nodes should be treated as compatibility nodes, not as recommended nodes for new workflow construction.

---

## Choosing the Right LoRA Loader

| Need | Recommended Node |
|---|---|
| Simple dynamic MODEL-only LoRA stack | JLC LoRA Loader - Multi Model |
| Simple dynamic MODEL+CLIP LoRA stack | JLC LoRA Loader - Multi-Model / CLIP |
| MODEL-only LoRA stack with one shared block profile | JLC LoRA Loader - Multi-Model / Shared Block Weight |
| MODEL+CLIP stack with one shared MODEL block profile | JLC LoRA Loader - Multi Model / Shared Block Weight + CLIP |
| MODEL-only stack with different block profile per LoRA | JLC LoRA Loader - Multi-Model / Block Weight |
| MODEL+CLIP stack with different MODEL block profile per LoRA | JLC LoRA Loader - Multi-Model / CLIP + Block Weight |
| Opening older saved workflows that used retained fixed-stack class types | Legacy compatibility wrappers |

A simple rule:

```text
Use plain loaders first.
Use shared block-weight loaders when all active LoRAs should share one block profile.
Use per-slot block-weight loaders only when each LoRA needs its own block profile.
Use MODEL+CLIP variants only when CLIP/text-encoder patching is desired.
```

---

## Example Workflows

Detailed LoRA workflow examples will be added shortly.

Planned examples:

- basic dynamic MODEL-only LoRA stack;
- MODEL+CLIP dynamic LoRA stack;
- shared block-weight LoRA stack;
- per-slot block-weight LoRA stack;
- compact showcase workflow combining dynamic LoRA loading with the JLC Seed Generator and ControlNet nodes.

Workflow will be added shortly.

---

## Notes for Advanced Users

### Sequential application order

Active LoRAs are applied sequentially in visible slot order. If two LoRAs interact strongly, their order may matter.

### Hidden values

Hidden slot values remain serialized. This is intentional and allows temporary simplification of the visible UI without destroying prior slot setup.

### `slot_count` is authoritative

A hidden slot is ignored even if it contains a selected LoRA and nonzero strengths.

### MODEL-only filtering

MODEL-only nodes filter clearly text-encoder / CLIP-side keys from LoRA files before MODEL-only application. Ambiguous keys are preserved so unexpected MODEL-side issues can still surface normally.

### Negative strengths

Strength widgets allow negative values. Negative LoRA strengths are an advanced technique and should be used deliberately.

### CLIP and block weights

Block vectors affect only MODEL/UNet patch weights. CLIP/text-encoder patching, when present, uses the ordinary CLIP strength for that slot.
