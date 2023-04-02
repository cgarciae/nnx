import contextlib
import dataclasses
import threading
import typing as tp
from types import MappingProxyType

import jax
import jax.tree_util as jtu

from refx import tracers
from nnx.rng_stream import RngStream
import refx

KeyArray = jax.random.KeyArray


class Scope:
    __slots__ = ("_rng_streams", "_flags")

    def __init__(
        self,
        rng_streams: tp.Mapping[tp.Hashable, RngStream],
        flags: tp.Mapping[str, tp.Hashable],
    ):
        self._rng_streams = MappingProxyType(rng_streams)
        self._flags = MappingProxyType(flags)

    @classmethod
    def empty(cls) -> "Scope":
        return Scope({}, {})

    @property
    def rng_streams(self) -> tp.Mapping[tp.Hashable, RngStream]:
        return self._rng_streams

    @property
    def flags(self) -> tp.Mapping[str, tp.Hashable]:
        return self._flags

    def fork(self) -> "Scope":
        rng_streams = {k: v.fork() for k, v in self._rng_streams.items()}
        return Scope(rng_streams, self._flags)

    def unsafe_trace_update(self):
        for rng_stream in self._rng_streams.values():
            rng_stream._jax_trace = tracers.current_jax_trace()
            rng_stream._refx_trace = tracers.current_refx_trace()


def _scope_flatten_with_keys(scope: Scope):
    return ((jtu.GetAttrKey("rng_streams"), scope._rng_streams.copy()),), scope._flags


def _scope_unflatten(flags, rng_streams):
    return Scope(rng_streams, flags)


jtu.register_pytree_with_keys(Scope, _scope_flatten_with_keys, _scope_unflatten)


@dataclasses.dataclass
class _ScopeContext(threading.local):
    scope_stack: tp.List[Scope] = dataclasses.field(
        default_factory=lambda: [Scope.empty()]
    )


_CONTEXT = _ScopeContext()


def current_scope() -> Scope:
    return _CONTEXT.scope_stack[-1]


@contextlib.contextmanager
def set_scope(
    rng_keys_or_scope: tp.Union[tp.Mapping[tp.Hashable, KeyArray], Scope],
    flags: tp.Optional[tp.Mapping[str, tp.Hashable]] = None,
):
    if isinstance(rng_keys_or_scope, Scope):
        if flags is not None:
            raise ValueError("Cannot set flags when passing a Scope")
        scope = rng_keys_or_scope
    else:
        if flags is None:
            flags = {}
        rng_streams = {k: RngStream(v) for k, v in rng_keys_or_scope.items()}
        scope = Scope(rng_streams, flags)
    context = _CONTEXT
    context.scope_stack.append(scope)
    try:
        yield
    finally:
        context.scope_stack.pop()


def fork_scope(flags: tp.Optional[tp.Mapping[str, tp.Hashable]] = None):
    return set_scope(current_scope().fork(), flags)


def make_rng(collection: tp.Hashable) -> KeyArray:
    scope = current_scope()

    if collection not in scope.rng_streams:
        raise ValueError(f"Unknown collection: {collection}")

    return scope.rng_streams[collection].next()


def get_flag(name: str) -> tp.Hashable:
    scope = current_scope()
    if name not in scope.flags:
        raise ValueError(f"Unknown flag: {name}")
    return scope.flags[name]
