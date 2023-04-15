import contextlib
import dataclasses
import threading
import typing as tp
from types import MappingProxyType

import jax
import jax.tree_util as jtu
import refx
from refx import tracers

from nnx import utils
from nnx.rng_stream import RngStream

KeyArray = jax.random.KeyArray


class Scope:
    __slots__ = ("_rng_streams", "_flags")

    def __init__(
        self,
        rng_streams: tp.Mapping[str, RngStream],
        flags: tp.Mapping[str, tp.Hashable],
    ):
        self._rng_streams = MappingProxyType(rng_streams)
        self._flags = MappingProxyType(flags)

    @classmethod
    def empty(cls) -> "Scope":
        return Scope({}, {})

    @classmethod
    def from_keys_and_flags(
        cls, keys: tp.Mapping[str, KeyArray], flags: tp.Mapping[str, tp.Hashable]
    ) -> "Scope":
        rng_streams = {k: RngStream(v) for k, v in keys.items()}
        return Scope(rng_streams, flags)

    @property
    def rng_streams(self) -> tp.Mapping[str, RngStream]:
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

    def copy(self) -> "Scope":
        return Scope(self._rng_streams, self._flags)


def _scope_flatten_with_keys(scope: Scope):
    return ((jtu.GetAttrKey("rng_streams"), scope._rng_streams.copy()),), scope._flags


def _scope_unflatten(flags, children):
    return Scope(children[0], flags)


jtu.register_pytree_with_keys(Scope, _scope_flatten_with_keys, _scope_unflatten)


@dataclasses.dataclass
class _ScopeContext(threading.local):
    scope_stack: tp.List[Scope] = dataclasses.field(
        default_factory=lambda: [Scope.empty()]
    )


_CONTEXT = _ScopeContext()


def current_scope() -> Scope:
    return _CONTEXT.scope_stack[-1]


@tp.overload
def scope(
    rngs: tp.Optional[tp.Mapping[tp.Hashable, KeyArray]] = None,
    /,
    *,
    flags: tp.Optional[tp.Mapping[str, tp.Hashable]] = None,
    trace: tp.Optional[tracers.MainTrace] = None,
    mutable: tp.Optional[tp.Callable[[tp.Hashable], bool]] = None,
) -> tp.ContextManager[None]:
    ...


@tp.overload
def scope(
    scope: Scope,
    /,
    *,
    trace: tp.Optional[tracers.MainTrace] = None,
    mutable: tp.Optional[tp.Callable[[tp.Hashable], bool]] = None,
) -> tp.ContextManager[None]:
    ...


@contextlib.contextmanager
def scope(
    rngs_or_scope: tp.Union[tp.Mapping[str, KeyArray], Scope, None] = None,
    /,
    *,
    flags: tp.Optional[tp.Mapping[str, tp.Hashable]] = None,
    trace: tp.Optional[tracers.MainTrace] = None,
    mutable: tp.Optional[tp.Callable[[tp.Hashable], bool]] = None,
):
    if isinstance(rngs_or_scope, Scope):
        if flags is not None:
            raise ValueError("Cannot set flags when passing a Scope")
        scope = rngs_or_scope.copy()
    else:
        if flags is None:
            flags = current_scope().flags
        if rngs_or_scope is None:
            rngs_or_scope = current_scope().rng_streams

        scope = Scope.from_keys_and_flags(rngs_or_scope, flags)
    _CONTEXT.scope_stack.append(scope)

    _contexts = []

    if trace is not None:
        _contexts.append(tracers.refx_trace(trace))

    if mutable is not None:
        _contexts.append(refx.mutable(mutable))

    try:
        with utils.contexts(*_contexts):
            scope.unsafe_trace_update()
            yield
    finally:
        _CONTEXT.scope_stack.pop()


def fork_scope():
    return scope(current_scope().fork())


def make_rng(collection: tp.Hashable) -> KeyArray:
    scope = current_scope()
    if collection not in scope.rng_streams:
        raise ValueError(f"Unknown collection: {collection}")
    return scope.rng_streams[collection].next()


def get_flag(name: str, default: tp.Any = dataclasses.MISSING) -> tp.Hashable:
    scope = current_scope()
    if name not in scope.flags:
        if default is dataclasses.MISSING:
            raise ValueError(f"Unknown flag: {name}")
        return default
    return scope.flags[name]


def all_mutable(_):
    return True


def init(
    rngs: tp.Union[KeyArray, tp.Dict[tp.Hashable, KeyArray], None] = None,
    /,
    *,
    mutable: tp.Callable[[tp.Hashable], bool] = all_mutable,
    flags: tp.Optional[tp.Dict[str, tp.Hashable]] = None,
    trace: tp.Optional[tracers.MainTrace] = None,
) -> tp.ContextManager[None]:
    if rngs is None:
        rngs = {}
    elif isinstance(rngs, jax.Array):
        rngs = {"params": rngs}

    if flags is None:
        flags = {}

    flags["initializing"] = True

    return scope(rngs, flags=flags, trace=trace, mutable=mutable)


def apply(
    rngs: tp.Union[KeyArray, tp.Dict[tp.Hashable, KeyArray], None] = None,
    /,
    *,
    mutable: tp.Callable[[tp.Hashable], bool] = all_mutable,
    flags: tp.Optional[tp.Dict[str, tp.Hashable]] = None,
    trace: tp.Optional[tracers.MainTrace] = None,
) -> tp.ContextManager[None]:
    if rngs is None:
        rngs = {}
    elif isinstance(rngs, jax.Array):
        rngs = {"dropout": rngs}

    if flags is None:
        flags = {}

    flags["initializing"] = False

    return scope(rngs, flags=flags, trace=trace, mutable=mutable)
