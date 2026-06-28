# TODO — JLC Seed Generator: Randomize Replay

Location suggestion:
`nodes/util_nodes/TODO_seed_generator_randomize_replay.md`

## Future enhancement

Add an optional replay mechanism for `randomize` mode so multi-run random seed trials can be repeated without relying on ComfyUI's internal random-number generation behavior.

## Current behavior

The JLC Seed Generator uses ComfyUI's native `control after generate` behavior while preserving the visible starting seed in the node UI. The frontend display shows the last seed actually used.

This works cleanly for:

- `fixed`
- `increment`
- `decrement`

These modes are deterministic from the visible starting seed.

## Proposed behavior for `randomize`

During a queued execution, record the actual seeds reported by the backend/frontend sequence:

```text
randomize run, count 4:
  seed A
  seed B
  seed C
  seed D
```

On a rerun where the visible input seed has not changed, optionally replay the stored sequence:

```text
replay run, count 4:
  seed A
  seed B
  seed C
  seed D
```

## Possible UI ideas

Keep this optional and off by default.

Potential controls:

```text
randomize_replay: off / record / replay
```

or:

```text
repeat_last_random_sequence: true / false
```

## Notes

This should only apply to `randomize` mode. It is unnecessary for `fixed`, `increment`, or `decrement`.

The goal is repeatable randomized parameter trials without needing to seed or reverse-engineer ComfyUI's internal random seed generation.
