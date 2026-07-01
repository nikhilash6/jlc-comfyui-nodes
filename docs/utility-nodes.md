# Utility Nodes

This chapter covers the JLC utility-node family:

- [JLC Seed Generator](#jlc-seed-generator)
- [Future Seed Generator Direction: Randomize Replay](#future-seed-generator-direction-randomize-replay)
- [JLC Stage Boundary VRAM Cleanup](#jlc-stage-boundary-vram-cleanup)
- [Choosing the Right Utility Node](#choosing-the-right-utility-node)
- [Example Workflows](#example-workflows)

These nodes are not image-generation algorithms by themselves. They are workflow-support nodes intended to make larger ComfyUI graphs easier to run, repeat, debug, or stage.

---

## JLC Seed Generator

**JLC Seed Generator** is a shared seed source for workflows where multiple samplers or multiple inference stages should use the same seed during one full prompt execution.

Typical use case:

```text
JLC Seed Generator
    ├─ seed_int → KSampler seed, stage 1
    └─ seed_int → KSampler seed, stage 2
```

This is useful when:

- one seed should feed multiple KSamplers;
- stage 1 creates a latent;
- stage 2 partially denoises the same latent;
- you want parameter trials without losing track of the starting seed.

### Outputs

The node returns two seed outputs:

| Output | Purpose |
|---|---|
| `seed` | Small SEED-style dictionary, e.g. `{"seed": 12345}`. |
| `seed_int` | Plain integer seed for nodes that expect an INT seed input. |

In most ordinary ComfyUI seed sockets, `seed_int` is the convenient output.

---

## Stable base seed and last-used display

The node uses ComfyUI's native `control_after_generate` seed behavior, but changes the user-facing display behavior.

A normal seed widget often mutates the visible seed after a queue submission. For example:

```text
seed = 1
control after generate = increment
queue count = 4
```

The visible widget may advance as queued prompts are prepared.

**JLC Seed Generator** instead keeps the visible seed input as the user's stable base seed, while the companion frontend display reports the seed actually used by the executed prompt.

Conceptually:

```text
visible base seed: 1

queued prompt 1 uses seed 1
queued prompt 2 uses seed 2
queued prompt 3 uses seed 3
queued prompt 4 uses seed 4

visible seed input is restored to 1
display reports the last seed actually used
```

The intent is to make parameter trials easier. You can queue several variations, abort, adjust parameters, and still see the original seed you started from.

### Display spacer

The node includes a harmless STRING widget row reserved for the frontend seed panel.

If the frontend script is available, it turns that row into a display panel. If the frontend script is unavailable, it remains an inert separator-like text field and is ignored by the backend.

---

## Future Seed Generator Direction: Randomize Replay

The current seed behavior works cleanly for:

```text
fixed
increment
decrement
```

Those modes are deterministic from the visible base seed.

The intended future improvement is an optional replay mechanism for `randomize` mode.

### Current randomize limitation

In `randomize` mode, ComfyUI may generate a different sequence of random seeds for a queued run. The JLC Seed Generator can display the seed actually used, but it does not yet replay an entire randomized sequence.

### Planned direction

The intended future feature is optional and off by default.

During a queued randomized trial, the node/frontend sequence could record the actual seeds used:

```text
randomize run, count 4:
  seed A
  seed B
  seed C
  seed D
```

On a rerun where the visible input seed has not changed, an optional replay mode could reuse the stored sequence:

```text
replay run, count 4:
  seed A
  seed B
  seed C
  seed D
```

Possible future controls might look like:

```text
randomize_replay: off / record / replay
```

or:

```text
repeat_last_random_sequence: true / false
```

### Why this matters

The long-term goal is repeatable randomized parameter trials.

That means a user could explore randomized seeds across a batch, then rerun the same randomized sequence after changing another parameter, without relying on or reverse-engineering ComfyUI's internal random-number behavior.

This planned feature should apply only to `randomize` mode. It is unnecessary for `fixed`, `increment`, and `decrement`.

---

## JLC Stage Boundary VRAM Cleanup

**JLC Stage Boundary VRAM Cleanup** is an experimental latent-passthrough cleanup node for advanced multi-stage workflows.

It is intended for workflows where one stage uses heavy model objects to produce a latent, then a later stage should run with a different model family or reduced resident-memory pressure.

Typical use case:

```text
Stage 1:
    large base model, inpaint model, or ControlNet stack
    ↓
    latent

Boundary:
    JLC Stage Boundary VRAM Cleanup
    ↓
    same latent passed through

Stage 2:
    different model family or partial denoising pass
```

The node returns the same `LATENT` it receives. Its purpose is side-effect cleanup at a deliberate stage boundary.

### Experimental warning

This node is experimental.

It may affect model residency, reload behavior, execution time, and VRAM usage in ways that depend on ComfyUI's current model-management internals.

ComfyUI remains the authority for model lifecycle management. This node should be treated as a best-effort helper, not as a guaranteed VRAM reset.

Use it only when the graph is structured so that the upstream model objects are no longer needed after the latent passthrough point.

---

## Cleanup targets

The robust targets are:

- a connected ComfyUI `MODEL` object and its clones/additional models;
- all currently loaded ComfyUI models, when explicitly requested;
- JLC-managed ControlNet resident cache entries;
- all JLC-managed resident cache entries, when explicitly requested;
- final best-effort Python/CUDA allocator cleanup.

It is intentionally not a generic CLIP/VAE cleanup node.

### Main controls

| Input | Purpose |
|---|---|
| `latent` | Passthrough latent that triggers the cleanup point. |
| `unload_connected_model` | Try to unload the connected optional MODEL and its clones/additional models. |
| `evict_jlc_controlnet_cache` | Evict JLC-managed ControlNet cache entries. |
| `evict_all_jlc_model_cache` | Evict all JLC-managed resident cache entries. |
| `unload_all_comfy_models` | Ask ComfyUI to unload all resident models. This is the broadest ComfyUI-side cleanup option. |
| `clear_cuda_allocator` | Run final best-effort allocator cleanup. |
| `safe_cleanup` | Use the safer cleanup path when supported by JLC cache helpers. |
| `all_devices` | Apply connected-model unload across devices when supported by ComfyUI. |
| `verbose` | Print cleanup status messages. |
| `model` | Optional connected MODEL to target for unload. |

### Execution behavior

The node intentionally forces execution when it is on an active graph path, because cleanup is a side effect. It should not be optimized away simply because the latent is cached.

### Practical guidance

Use the narrowest cleanup that solves the problem.

Start with:

```text
unload_connected_model = true
clear_cuda_allocator = true
```

Then add broader options only when needed:

```text
evict_jlc_controlnet_cache = true
```

or, for more aggressive cleanup:

```text
unload_all_comfy_models = true
evict_all_jlc_model_cache = true
```

The broad options may cause later nodes to reload models, which can increase execution time.

---

## Choosing the Right Utility Node

| Need | Recommended Node |
|---|---|
| Feed the same seed into multiple samplers or stages | JLC Seed Generator |
| Keep the visible seed stable while viewing the last seed actually used | JLC Seed Generator |
| Prepare for future repeatable randomized seed trials | JLC Seed Generator, with planned randomize replay enhancement |
| Pass a latent across a deliberate stage boundary while trying to free selected model objects | JLC Stage Boundary VRAM Cleanup |
| Force a guaranteed complete VRAM reset | Not guaranteed by these nodes; restart ComfyUI if a true reset is required |

---

## Example Workflows

Detailed utility workflow examples will be added shortly.

Planned examples:

- shared seed feeding two samplers;
- seed generator in a multi-stage partial-denoise workflow;
- stage-boundary cleanup between a ControlNet-heavy first stage and a second sampler;
- compact showcase workflow combining the Seed Generator, dynamic LoRA loaders, ControlNet Aux preprocessing, and ControlNet composition/orchestration.

Workflow will be added shortly.

---

## Notes for Advanced Users

### Seed dictionary vs. integer output

The `seed` output is a small dictionary for compatibility with seed-style consumers. The `seed_int` output is a plain integer and is usually the easiest connection for standard sampler seed fields.

### Randomize replay is not implemented yet

The randomize replay section documents the planned direction. It is included here so the intent of the current seed-display design is clear, but the replay feature itself is not part of the current implementation.

### Stage cleanup is not a magic memory eraser

The VRAM cleanup node can request targeted unloads and allocator cleanup, but model residency remains dependent on ComfyUI internals, active graph references, backend behavior, and selected options.

### Avoid using cleanup too early

Place the cleanup node only after the upstream model objects are truly no longer needed. If the graph still needs those objects later, ComfyUI may reload them or the workflow may behave unexpectedly.

### Verbose mode

Verbose mode is useful while designing workflows because it prints what cleanup actions were requested and how long the cleanup pass took.
