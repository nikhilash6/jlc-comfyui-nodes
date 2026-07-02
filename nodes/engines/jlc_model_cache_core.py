"""
JLC Shared Model Residency Cache Core
-------------------------------------

- JLC ComfyUI Nodes Collection
  - This module is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/jlc-comfyui-nodes

- Engine Purpose
    - The **JLC Shared Model Residency Cache Core** provides lightweight,
      process-local residency management for heavyweight model objects used by
      JLC ComfyUI custom nodes.

    - It is intentionally generic and collection-level. It is designed to be
      reused by dynamic/internal-loader nodes such as:
            • ControlNet loaders / orchestrators
            • VAE helpers
            • segmenters
            • upscalers
            • CLIP vision / IPAdapter helpers
            • other heavyweight model-loading JLC nodes

    - It supports:
            • stable cache-key construction
            • reuse of already-loaded objects
            • family-aware LRU capacity limits
            • optional total cache capacity
            • optional unload callbacks
            • Python reference cleanup
            • best-effort CUDA allocator cleanup
            • fast vs safe eviction modes
            • lightweight diagnostics

- Design Notes
    - This file owns only cache/residency policy. It must not own model loading
      logic, ControlNet composition logic, conditioning math, or sampler-facing
      behavior.

    - Cached objects may be model instances, dictionaries, tuples, or small
      engine-specific bundles.

    - Eviction should be performed during node execution/model preparation,
      before prepared per-slot copies are injected into sampler-facing objects.
      Do not evict resident cache objects from inside per-sampling callbacks
      such as ControlNet get_control paths.

    - For ControlNet use, the intended default is a bounded family cache of
      roughly four resident ControlNet objects:
            key = make_controlnet_cache_key(full_path)
            obj = get_or_load_model(... family="controlnet" ...)
      Per-slot execution should still use independent .copy() isolation exactly
      where the orchestrator already does so.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

import gc
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

from ...jlc_custom_nodes_versions import JLC_MODEL_CACHE_CORE_VERSION

MANIFEST = {
    "name": "JLC Shared Model Residency Cache Core",
    "version": JLC_MODEL_CACHE_CORE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Shared process-local cache/residency helper for heavyweight JLC "
        "ComfyUI node models. Provides stable keys, bounded family-aware LRU "
        "reuse, optional unload callbacks, CUDA cleanup, safe/fast eviction "
        "modes, and lightweight diagnostics while keeping model loading and "
        "node algorithms separate."
    ),
}


try:
    import torch
except Exception:
    # Keep import safe during metadata inspection, dependency scans, or CPU-only
    # environments. CUDA cleanup helpers become no-ops when torch is unavailable.
    torch = None


UnloadFn = Callable[[Any], None]
LoaderFn = Callable[[], Any]


# -------------------------------------------------------------------------
# Environment helpers
# -------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _env_first_int(names: Iterable[str], default: int) -> int:
    for name in names:
        if name in os.environ:
            return _env_int(name, default)
    return int(default)


# Total process-local entry cap. Family caps usually matter more.
# Keep this comfortably above the ControlNet default so future families can opt in.
_DEFAULT_MAX_LOADED_TOTAL = max(
    0,
    _env_first_int(
        (
            "JLC_MODEL_CACHE_MAX_LOADED_TOTAL",
            "JLC_CACHE_MAX_LOADED_TOTAL",
        ),
        32,
    ),
)

# ControlNet default is intentionally around 2 resident base models.
_DEFAULT_CONTROLNET_CAPACITY = max(
    0,
    _env_first_int(
        (
            "JLC_CONTROLNET_CACHE_SIZE",
            "JLC_MODEL_CACHE_MAX_CONTROLNET",
            "JLC_CACHE_MAX_CONTROLNET",
        ),
        2,
    ),
)

_SAFE_EVICTION_COOLDOWN_SEC = max(
    0.0,
    _env_float("JLC_MODEL_CACHE_SAFE_EVICTION_COOLDOWN_SEC", 1.25),
)
_FAST_EVICTION_COOLDOWN_SEC = max(
    0.0,
    _env_float("JLC_MODEL_CACHE_FAST_EVICTION_COOLDOWN_SEC", 0.0),
)
_VERBOSE_CACHE = _env_bool("JLC_MODEL_CACHE_VERBOSE", False)


# -------------------------------------------------------------------------
# Global process-local state
# -------------------------------------------------------------------------


_CACHE_LOCK = threading.RLock()
_MODEL_CACHE: Dict[str, "CacheEntry"] = {}
_MAX_LOADED_TOTAL = _DEFAULT_MAX_LOADED_TOTAL
_FAMILY_CAPACITY: Dict[str, int] = {
    "controlnet": _DEFAULT_CONTROLNET_CAPACITY,
}


@dataclass
class CacheEntry:
    """A single resident cache object."""

    key: str
    obj: Any
    family: str
    model_path: str = "none"
    role: str = "model"
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    revision: str = "default"
    variant: str = "default"
    unload_fn: Optional[UnloadFn] = None
    keep: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def touch(self) -> None:
        self.last_used_at = time.time()
        self.hit_count += 1


# -------------------------------------------------------------------------
# Normalization and key construction
# -------------------------------------------------------------------------


def normalize_cache_part(value: Any) -> str:
    """Normalize one non-path key component."""

    if value is None:
        return "none"
    text = str(value).strip()
    return text if text else "none"


def normalize_family(family: Any) -> str:
    return normalize_cache_part(family).lower()


def normalize_role(role: Any) -> str:
    return normalize_cache_part(role).lower()


def normalize_model_path(model_path: Any) -> str:
    """
    Normalize a filesystem model path for stable cache keys.

    For path-like values, this expands user/environment variables, converts to
    an absolute path, normalizes separators, and applies os.path.normcase so
    Windows paths compare case-insensitively.

    For non-path identifiers such as HF repo ids, this leaves the identifier
    mostly intact.
    """

    if model_path is None:
        return "none"

    text = str(model_path).strip()
    if not text:
        return "none"

    expanded = os.path.expandvars(os.path.expanduser(text))

    # Treat strings with path separators, drive roots, or existing filesystem
    # targets as paths. Plain repo ids / names stay as stable text identifiers.
    has_sep = os.path.sep in expanded
    has_altsep = bool(os.path.altsep and os.path.altsep in expanded)
    looks_abs = os.path.isabs(expanded)
    exists = os.path.exists(expanded)

    if has_sep or has_altsep or looks_abs or exists:
        try:
            return os.path.normcase(os.path.abspath(os.path.normpath(expanded)))
        except Exception:
            return os.path.normcase(os.path.normpath(expanded))

    return expanded


def make_cache_key(
    *,
    family: str,
    model_path: Any = "none",
    role: str = "model",
    device: str = "auto",
    dtype: str = "auto",
    quantization: str = "none",
    revision: str = "default",
    variant: str = "default",
) -> str:
    """
    Build a stable generic cache key.

    Include fields that materially affect the resident object. This is useful
    for future families where device, dtype, quantization, or revision produce
    different Python objects.
    """

    parts = [
        normalize_role(role),
        normalize_family(family),
        normalize_model_path(model_path),
        normalize_cache_part(device).lower(),
        normalize_cache_part(dtype).lower(),
        normalize_cache_part(quantization).lower(),
        normalize_cache_part(revision),
        normalize_cache_part(variant),
    ]
    return "::".join(parts)


def make_path_cache_key(*, family: str, model_path: Any) -> str:
    """
    Build a compact family/path key.

    This is the preferred key shape for ComfyUI model objects whose resident
    identity is fully determined by the normalized full path.
    """

    return f"{normalize_family(family)}::{normalize_model_path(model_path)}"


def make_controlnet_cache_key(model_path: Any) -> str:
    """
    Build the standard JLC ControlNet cache key.

    Example:
        controlnet::c:\\path\\to\\controlnet.safetensors
    """

    return make_path_cache_key(family="controlnet", model_path=model_path)


# -------------------------------------------------------------------------
# Capacity configuration
# -------------------------------------------------------------------------


def set_max_loaded_total(max_loaded_total: int) -> None:
    """Set the optional total entry capacity for all cache families."""

    global _MAX_LOADED_TOTAL

    value = int(max_loaded_total)
    if value < 0:
        raise ValueError("max_loaded_total must be >= 0")

    with _CACHE_LOCK:
        _MAX_LOADED_TOTAL = value
        _enforce_total_capacity_locked(protected_key=None)


def get_max_loaded_total() -> int:
    with _CACHE_LOCK:
        return _MAX_LOADED_TOTAL


def set_family_capacity(family: str, max_loaded: Optional[int]) -> None:
    """
    Set the maximum resident entries for one family.

    Pass None to remove the explicit family cap. Pass 0 to evict/prevent all
    non-keep entries for that family.
    """

    family_norm = normalize_family(family)

    with _CACHE_LOCK:
        if max_loaded is None:
            _FAMILY_CAPACITY.pop(family_norm, None)
        else:
            value = int(max_loaded)
            if value < 0:
                raise ValueError("family capacity must be >= 0 or None")
            _FAMILY_CAPACITY[family_norm] = value
            _enforce_family_capacity_locked(family_norm, protected_key=None)


def get_family_capacity(family: str) -> Optional[int]:
    with _CACHE_LOCK:
        return _FAMILY_CAPACITY.get(normalize_family(family))


def get_controlnet_cache_capacity() -> int:
    value = get_family_capacity("controlnet")
    return int(value if value is not None else 0)


def set_controlnet_cache_capacity(max_loaded: int) -> None:
    set_family_capacity("controlnet", int(max_loaded))


# Compatibility aliases for node code that prefers CaptionForge-like names.
def set_max_loaded_models(max_loaded_models: int) -> None:
    set_max_loaded_total(max_loaded_models)


def get_max_loaded_models() -> int:
    return get_max_loaded_total()


# -------------------------------------------------------------------------
# Read/reuse/register API
# -------------------------------------------------------------------------


def get_cached_model(key: str) -> Optional[Any]:
    """Return a cached object by key, touching LRU state if present."""

    with _CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is None:
            return None

        entry.touch()
        print(f"[JLC Model Cache] Reusing cached model: {key}")
        return entry.obj


def has_cached_model(key: str) -> bool:
    with _CACHE_LOCK:
        return key in _MODEL_CACHE


def register_model(
    key: str,
    obj: Any,
    *,
    family: str,
    model_path: Any = "none",
    role: str = "model",
    device: str = "auto",
    dtype: str = "auto",
    quantization: str = "none",
    revision: str = "default",
    variant: str = "default",
    unload_fn: Optional[UnloadFn] = None,
    keep: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    max_loaded_for_family: Optional[int] = None,
) -> Any:
    """
    Register a resident object and enforce capacity.

    The object can be a model, a tuple/dict containing model-related objects,
    or a node-specific bundle dataclass.
    """

    family_norm = normalize_family(family)

    with _CACHE_LOCK:
        if max_loaded_for_family is not None:
            set_family_capacity(family_norm, int(max_loaded_for_family))

        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            existing.obj = obj
            existing.family = family_norm
            existing.model_path = normalize_model_path(model_path)
            existing.role = normalize_role(role)
            existing.device = normalize_cache_part(device).lower()
            existing.dtype = normalize_cache_part(dtype).lower()
            existing.quantization = normalize_cache_part(quantization).lower()
            existing.revision = normalize_cache_part(revision)
            existing.variant = normalize_cache_part(variant)
            existing.unload_fn = unload_fn
            existing.keep = bool(keep)
            existing.metadata = dict(metadata or {})
            existing.touch()
        else:
            _MODEL_CACHE[key] = CacheEntry(
                key=key,
                obj=obj,
                family=family_norm,
                model_path=normalize_model_path(model_path),
                role=normalize_role(role),
                device=normalize_cache_part(device).lower(),
                dtype=normalize_cache_part(dtype).lower(),
                quantization=normalize_cache_part(quantization).lower(),
                revision=normalize_cache_part(revision),
                variant=normalize_cache_part(variant),
                unload_fn=unload_fn,
                keep=bool(keep),
                metadata=dict(metadata or {}),
            )

        print(f"[JLC Model Cache] Registered model: {key}")
        _enforce_family_capacity_locked(family_norm, protected_key=key)
        _enforce_total_capacity_locked(protected_key=key)
        return obj


def get_or_load_model(
    key: str,
    loader_fn: LoaderFn,
    *,
    family: str,
    model_path: Any = "none",
    role: str = "model",
    device: str = "auto",
    dtype: str = "auto",
    quantization: str = "none",
    revision: str = "default",
    variant: str = "default",
    unload_fn: Optional[UnloadFn] = None,
    keep: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    policy: str = "lru_family_capacity",
    max_loaded_for_family: Optional[int] = None,
) -> Any:
    """
    Return a cached object or load/register it.

    This is the simplest integration point for internal-loader nodes.
    """

    cached = get_cached_model(key)
    if cached is not None:
        return cached

    prepare_for_model_load(
        key,
        family=family,
        role=role,
        policy=policy,
        max_loaded_for_family=max_loaded_for_family,
    )

    obj = loader_fn()

    return register_model(
        key,
        obj,
        family=family,
        model_path=model_path,
        role=role,
        device=device,
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        variant=variant,
        unload_fn=unload_fn,
        keep=keep,
        metadata=metadata,
        max_loaded_for_family=max_loaded_for_family,
    )


# -------------------------------------------------------------------------
# Preparation / eviction API
# -------------------------------------------------------------------------


def prepare_for_model_load(
    key: str,
    *,
    family: str,
    role: str = "model",
    policy: str = "lru_family_capacity",
    max_loaded_for_family: Optional[int] = None,
) -> None:
    """
    Call immediately before loading a heavyweight model.

    Policies:
    - none:
        Do nothing before load.
    - keep_this_model:
        Do not evict immediately. Capacity is still enforced after register.
    - lru_family_capacity:
        If this key is not already cached, evict least-recently-used entries
        from the same family until there is room for one new entry.
    - lru_family_capacity_safe:
        Same as lru_family_capacity, with safe CUDA cleanup cooldown.
    - evict_other_family:
        Evict other entries in the same family before loading this key.
    - evict_other_family_safe:
        Safe version of evict_other_family.
    - evict_other_role:
        Evict other entries with the same role before loading this key.
    - evict_other_role_safe:
        Safe version of evict_other_role.
    - evict_all:
        Evict all other non-keep entries.
    - evict_all_safe:
        Safe version of evict_all.

    For ControlNet internal loaders, use the default lru_family_capacity policy
    with family="controlnet" and max_loaded_for_family around 4.
    """

    family_norm = normalize_family(family)
    role_norm = normalize_role(role)
    policy_norm = (policy or "").strip().lower() or "lru_family_capacity"
    safe_mode = policy_norm.endswith("_safe")
    base_policy = policy_norm[:-5] if safe_mode else policy_norm

    with _CACHE_LOCK:
        if max_loaded_for_family is not None:
            set_family_capacity(family_norm, int(max_loaded_for_family))

        if key in _MODEL_CACHE:
            _MODEL_CACHE[key].touch()
            return

        if base_policy in {"none", "keep_this_model"}:
            return

        did_evict = False

        if base_policy == "lru_family_capacity":
            did_evict = _make_room_for_family_locked(
                family_norm,
                incoming_key=key,
                safe=safe_mode,
                reason=f"policy={policy_norm}",
            ) or did_evict
            did_evict = _make_room_for_total_locked(
                incoming_key=key,
                safe=safe_mode,
                reason=f"policy={policy_norm}:total",
            ) or did_evict

        elif base_policy == "evict_other_family":
            for existing_key, entry in list(_MODEL_CACHE.items()):
                if existing_key == key:
                    continue
                if entry.family == family_norm:
                    did_evict = _evict_locked(
                        existing_key,
                        reason=f"policy={policy_norm}",
                        safe=safe_mode,
                    ) or did_evict

        elif base_policy == "evict_other_role":
            for existing_key, entry in list(_MODEL_CACHE.items()):
                if existing_key == key:
                    continue
                if entry.role == role_norm:
                    did_evict = _evict_locked(
                        existing_key,
                        reason=f"policy={policy_norm}",
                        safe=safe_mode,
                    ) or did_evict

        elif base_policy == "evict_all":
            for existing_key in list(_MODEL_CACHE.keys()):
                if existing_key == key:
                    continue
                did_evict = _evict_locked(
                    existing_key,
                    reason=f"policy={policy_norm}",
                    safe=safe_mode,
                ) or did_evict

        else:
            raise ValueError(
                f"Unknown JLC model cache policy: {policy!r}. "
                "Expected one of: none, keep_this_model, lru_family_capacity, "
                "lru_family_capacity_safe, evict_other_family, "
                "evict_other_family_safe, evict_other_role, "
                "evict_other_role_safe, evict_all, evict_all_safe."
            )

        if did_evict:
            _cuda_cleanup(
                reason=f"post_prepare_for_model_load:{policy_norm}",
                synchronize=True,
                cooldown_sec=(
                    _SAFE_EVICTION_COOLDOWN_SEC
                    if safe_mode
                    else _FAST_EVICTION_COOLDOWN_SEC
                ),
            )


def evict_model(key: str, *, reason: str = "manual", safe: bool = False) -> bool:
    """Evict one cached object by key."""

    with _CACHE_LOCK:
        did_evict = _evict_locked(key, reason=reason, safe=safe)
        if did_evict:
            _cuda_cleanup(
                reason=f"post_evict_model:{reason}",
                synchronize=True,
                cooldown_sec=(
                    _SAFE_EVICTION_COOLDOWN_SEC
                    if safe
                    else _FAST_EVICTION_COOLDOWN_SEC
                ),
            )
        return did_evict


def evict_family(family: str, *, reason: str = "manual_family_evict", safe: bool = False) -> int:
    """Evict all non-keep entries for a family."""

    family_norm = normalize_family(family)
    count = 0

    with _CACHE_LOCK:
        for key, entry in list(_MODEL_CACHE.items()):
            if entry.family == family_norm:
                if _evict_locked(key, reason=reason, safe=safe):
                    count += 1

        if count:
            _cuda_cleanup(
                reason=f"post_evict_family:{family_norm}",
                synchronize=True,
                cooldown_sec=(
                    _SAFE_EVICTION_COOLDOWN_SEC
                    if safe
                    else _FAST_EVICTION_COOLDOWN_SEC
                ),
            )
        return count


def evict_role(role: str, *, reason: str = "manual_role_evict", safe: bool = False) -> int:
    """Evict all non-keep entries for a role."""

    role_norm = normalize_role(role)
    count = 0

    with _CACHE_LOCK:
        for key, entry in list(_MODEL_CACHE.items()):
            if entry.role == role_norm:
                if _evict_locked(key, reason=reason, safe=safe):
                    count += 1

        if count:
            _cuda_cleanup(
                reason=f"post_evict_role:{role_norm}",
                synchronize=True,
                cooldown_sec=(
                    _SAFE_EVICTION_COOLDOWN_SEC
                    if safe
                    else _FAST_EVICTION_COOLDOWN_SEC
                ),
            )
        return count


def unload_all(*, include_keep: bool = True, reason: str = "manual_unload_all", safe: bool = False) -> int:
    """Clear the cache."""

    count = 0

    with _CACHE_LOCK:
        for key in list(_MODEL_CACHE.keys()):
            entry = _MODEL_CACHE.get(key)
            if entry is None:
                continue
            if entry.keep and not include_keep:
                continue
            if _evict_locked(key, reason=reason, safe=safe):
                count += 1

        if count:
            _cuda_cleanup(
                reason=f"post_unload_all:{reason}",
                synchronize=True,
                cooldown_sec=(
                    _SAFE_EVICTION_COOLDOWN_SEC
                    if safe
                    else _FAST_EVICTION_COOLDOWN_SEC
                ),
            )
        return count


def unload_after_run(key: str, *, enabled: bool, safe: bool = False) -> None:
    """Convenience helper for explicit unload-after-run node policies."""

    if enabled:
        evict_model(
            key,
            reason="unload_after_run_safe" if safe else "unload_after_run",
            safe=safe,
        )


# -------------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------------


def cache_size(
    *,
    family: Optional[str] = None,
    role: Optional[str] = None,
    include_keep: bool = True,
) -> int:
    """Return number of resident entries matching optional filters."""

    family_norm = normalize_family(family) if family is not None else None
    role_norm = normalize_role(role) if role is not None else None

    with _CACHE_LOCK:
        count = 0
        for entry in _MODEL_CACHE.values():
            if not include_keep and entry.keep:
                continue
            if family_norm is not None and entry.family != family_norm:
                continue
            if role_norm is not None and entry.role != role_norm:
                continue
            count += 1
        return count


def cache_keys(
    *,
    family: Optional[str] = None,
    role: Optional[str] = None,
    include_keep: bool = True,
) -> list[str]:
    """Return cache keys matching optional filters."""

    family_norm = normalize_family(family) if family is not None else None
    role_norm = normalize_role(role) if role is not None else None

    with _CACHE_LOCK:
        keys = []
        for key, entry in _MODEL_CACHE.items():
            if not include_keep and entry.keep:
                continue
            if family_norm is not None and entry.family != family_norm:
                continue
            if role_norm is not None and entry.role != role_norm:
                continue
            keys.append(key)
        return keys


def cache_info() -> Dict[str, Any]:
    """Return lightweight cache diagnostics."""

    now = time.time()

    with _CACHE_LOCK:
        entries = []
        for key, entry in _MODEL_CACHE.items():
            entries.append(
                {
                    "key": key,
                    "family": entry.family,
                    "role": entry.role,
                    "model_path": entry.model_path,
                    "device": entry.device,
                    "dtype": entry.dtype,
                    "quantization": entry.quantization,
                    "revision": entry.revision,
                    "variant": entry.variant,
                    "keep": entry.keep,
                    "age_sec": round(now - entry.created_at, 3),
                    "idle_sec": round(now - entry.last_used_at, 3),
                    "hit_count": entry.hit_count,
                    "metadata": dict(entry.metadata or {}),
                }
            )

        entries.sort(key=lambda item: (item["family"], item["idle_sec"], item["key"]))

        return {
            "max_loaded_total": _MAX_LOADED_TOTAL,
            "loaded_count": len(_MODEL_CACHE),
            "family_capacity": dict(_FAMILY_CAPACITY),
            "safe_eviction_cooldown_sec": _SAFE_EVICTION_COOLDOWN_SEC,
            "fast_eviction_cooldown_sec": _FAST_EVICTION_COOLDOWN_SEC,
            "entries": entries,
        }


def print_cache_info(prefix: str = "[JLC Model Cache]") -> None:
    """Print compact diagnostics to the ComfyUI console."""

    info = cache_info()
    print(
        f"{prefix} loaded={info['loaded_count']} "
        f"max_total={info['max_loaded_total']} "
        f"family_capacity={info['family_capacity']}"
    )
    for entry in info["entries"]:
        print(
            f"{prefix} - {entry['key']} | family={entry['family']} "
            f"role={entry['role']} keep={entry['keep']} "
            f"idle={entry['idle_sec']}s hits={entry['hit_count']}"
        )


def set_safe_eviction_cooldown(seconds: float) -> None:
    global _SAFE_EVICTION_COOLDOWN_SEC
    _SAFE_EVICTION_COOLDOWN_SEC = max(0.0, float(seconds))


def get_safe_eviction_cooldown() -> float:
    return _SAFE_EVICTION_COOLDOWN_SEC


def set_fast_eviction_cooldown(seconds: float) -> None:
    global _FAST_EVICTION_COOLDOWN_SEC
    _FAST_EVICTION_COOLDOWN_SEC = max(0.0, float(seconds))


def get_fast_eviction_cooldown() -> float:
    return _FAST_EVICTION_COOLDOWN_SEC


# -------------------------------------------------------------------------
# Internal capacity helpers
# -------------------------------------------------------------------------


def _family_entries_locked(family_norm: str) -> list[CacheEntry]:
    return [entry for entry in _MODEL_CACHE.values() if entry.family == family_norm]


def _evictable_lru_entries_locked(
    *,
    family_norm: Optional[str] = None,
    protected_key: Optional[str] = None,
) -> list[CacheEntry]:
    entries = []
    for key, entry in _MODEL_CACHE.items():
        if protected_key is not None and key == protected_key:
            continue
        if entry.keep:
            continue
        if family_norm is not None and entry.family != family_norm:
            continue
        entries.append(entry)

    entries.sort(key=lambda entry: entry.last_used_at)
    return entries


def _make_room_for_family_locked(
    family_norm: str,
    *,
    incoming_key: Optional[str],
    safe: bool,
    reason: str,
) -> bool:
    capacity = _FAMILY_CAPACITY.get(family_norm)
    if capacity is None:
        return False

    # Need one free slot for incoming_key when it is not already cached.
    target_count = max(0, int(capacity) - 1)
    did_evict = False

    while len(_family_entries_locked(family_norm)) > target_count:
        candidates = _evictable_lru_entries_locked(
            family_norm=family_norm,
            protected_key=incoming_key,
        )
        if not candidates:
            break
        victim = candidates[0]
        did_evict = _evict_locked(victim.key, reason=reason, safe=safe) or did_evict

    return did_evict


def _make_room_for_total_locked(
    *,
    incoming_key: Optional[str],
    safe: bool,
    reason: str,
) -> bool:
    if _MAX_LOADED_TOTAL < 0:
        return False

    # Need one free slot for incoming_key when it is not already cached.
    target_count = max(0, int(_MAX_LOADED_TOTAL) - 1)
    did_evict = False

    while len(_MODEL_CACHE) > target_count:
        candidates = _evictable_lru_entries_locked(protected_key=incoming_key)
        if not candidates:
            break
        victim = candidates[0]
        did_evict = _evict_locked(victim.key, reason=reason, safe=safe) or did_evict

    return did_evict


def _enforce_family_capacity_locked(family_norm: str, protected_key: Optional[str]) -> None:
    capacity = _FAMILY_CAPACITY.get(family_norm)
    if capacity is None:
        return

    while len(_family_entries_locked(family_norm)) > int(capacity):
        candidates = _evictable_lru_entries_locked(
            family_norm=family_norm,
            protected_key=protected_key,
        )
        if not candidates:
            break
        victim = candidates[0]
        _evict_locked(victim.key, reason="family_capacity", safe=False)
        _cuda_cleanup(
            reason=f"post_family_capacity_evict:{family_norm}",
            synchronize=True,
            cooldown_sec=_FAST_EVICTION_COOLDOWN_SEC,
        )


def _enforce_total_capacity_locked(protected_key: Optional[str]) -> None:
    if _MAX_LOADED_TOTAL < 0:
        return

    while len(_MODEL_CACHE) > int(_MAX_LOADED_TOTAL):
        candidates = _evictable_lru_entries_locked(protected_key=protected_key)
        if not candidates:
            break
        victim = candidates[0]
        _evict_locked(victim.key, reason="total_capacity", safe=False)
        _cuda_cleanup(
            reason="post_total_capacity_evict",
            synchronize=True,
            cooldown_sec=_FAST_EVICTION_COOLDOWN_SEC,
        )


def _evict_locked(key: str, *, reason: str = "unspecified", safe: bool = False) -> bool:
    entry = _MODEL_CACHE.pop(key, None)
    if entry is None:
        return False

    print(f"[JLC Model Cache] Evicting model: {key} | reason={reason}")

    # Synchronize before tearing down CUDA-backed model objects. This gives
    # native CUDA paths a cleaner boundary before references are cleared.
    _cuda_synchronize(reason=f"pre_unload:{reason}", verbose=safe or _VERBOSE_CACHE)

    try:
        if entry.unload_fn is not None:
            entry.unload_fn(entry.obj)
    except Exception as exc:
        print(f"[JLC Model Cache] Warning: unload_fn failed for {key}: {exc}")

    try:
        entry.obj = None
    except Exception:
        pass

    del entry
    gc.collect()

    # A second synchronize catches any work triggered by custom unload hooks.
    _cuda_synchronize(reason=f"post_unload:{reason}", verbose=safe or _VERBOSE_CACHE)
    return True


# -------------------------------------------------------------------------
# CUDA cleanup helpers
# -------------------------------------------------------------------------


def _cuda_synchronize(*, reason: str = "", verbose: bool = False) -> None:
    if torch is None:
        return

    try:
        if torch.cuda.is_available():
            if verbose:
                print(f"[JLC Model Cache] CUDA synchronize start ({reason})")
            torch.cuda.synchronize()
            if verbose:
                print(f"[JLC Model Cache] CUDA synchronize done ({reason})")
    except Exception as exc:
        print(f"[JLC Model Cache] Warning: CUDA synchronize failed ({reason}): {exc}")


def cuda_cleanup(
    *,
    reason: str = "manual_cuda_cleanup",
    synchronize: bool = True,
    safe: bool = False,
) -> None:
    """Public best-effort Python/CUDA cleanup helper."""

    _cuda_cleanup(
        reason=reason,
        synchronize=synchronize,
        cooldown_sec=_SAFE_EVICTION_COOLDOWN_SEC if safe else _FAST_EVICTION_COOLDOWN_SEC,
    )


def _cuda_cleanup(
    *,
    reason: str = "cuda_cleanup",
    synchronize: bool = True,
    cooldown_sec: float = 0.0,
) -> None:
    """
    Clear Python and CUDA allocator leftovers after eviction.

    This is best-effort and intentionally defensive for Windows/CUDA setups.
    """

    gc.collect()

    if torch is None:
        if cooldown_sec > 0:
            time.sleep(float(cooldown_sec))
        return

    if synchronize:
        _cuda_synchronize(reason=f"pre_cleanup:{reason}", verbose=_VERBOSE_CACHE)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                # ipc_collect can fail harmlessly on some Windows/CUDA setups.
                pass
    except Exception as exc:
        print(f"[JLC Model Cache] Warning: CUDA cleanup failed ({reason}): {exc}")

    gc.collect()

    if synchronize:
        _cuda_synchronize(reason=f"post_cleanup:{reason}", verbose=_VERBOSE_CACHE)

    if cooldown_sec > 0:
        if _VERBOSE_CACHE:
            print(f"[JLC Model Cache] Cooldown {cooldown_sec:.2f}s ({reason})")
        time.sleep(float(cooldown_sec))
