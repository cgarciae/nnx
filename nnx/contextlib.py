import hashlib
import typing as tp
from types import MappingProxyType

import jax
import jax.tree_util as jtu

from nnx import errors, tracers

KeyArray = jax.Array
Counts = tp.Tuple[int, ...]


def _stable_hash(data: Counts) -> int:
    hash_str = " ".join(str(x) for x in data)
    _hash = hashlib.blake2s(hash_str.encode())
    hash_bytes = _hash.digest()
    # uint32 is represented as 4 bytes in big endian
    return int.from_bytes(hash_bytes[:4], byteorder="big")


class RngStream:
    __slots__ = ("_key", "_count", "_count_path", "_trace_state")

    def __init__(self, key: KeyArray, count: int = 0, count_path: Counts = ()):
        self._key = key
        self._count = count
        self._count_path = count_path
        self._trace_state = tracers.TraceState()

    @property
    def count(self) -> int:
        return self._count

    @property
    def count_path(self) -> Counts:
        return self._count_path

    def make_rng(self) -> jax.random.KeyArray:
        if not self._trace_state.is_valid():
            raise errors.TraceContextError(
                "Cannot use RngStream from a different trace level"
            )

        fold_data = _stable_hash(self._count_path + (self._count,))
        self._count += 1
        return jax.random.fold_in(self._key, fold_data)

    def fork(self) -> "RngStream":
        if not self._trace_state.is_valid():
            raise errors.TraceContextError(
                "Cannot use RngStream from a different trace level"
            )
        count_path = self._count_path + (self._count,)
        self._count += 1
        return RngStream(self._key, count_path=count_path)

    def split(self, n: int = 2) -> "RngStream":
        if not self._trace_state.is_valid():
            raise errors.TraceContextError(
                "Cannot use RngStream from a different trace level"
            )
        key = self.make_rng()
        key = jax.random.split(key, n)
        return RngStream(key)


def _rng_stream_flatten_with_keys(
    rng: RngStream,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[tp.Hashable, jax.random.KeyArray], ...],
    tp.Tuple[int, Counts],
]:
    return ((jtu.GetAttrKey("key"), rng._key),), (rng.count, rng.count_path)


def _rng_stream_unflatten(
    aux_data: tp.Tuple[int, Counts],
    children: tp.Tuple[jax.random.KeyArray, ...],
) -> RngStream:
    count, count_path = aux_data
    key = children[0]
    return RngStream(key, count, count_path)


def _rng_stream_flatten(rng: RngStream):
    return (rng._key,), (rng.count, rng.count_path)


jax.tree_util.register_pytree_with_keys(
    RngStream,
    _rng_stream_flatten_with_keys,
    _rng_stream_unflatten,
    flatten_func=_rng_stream_flatten,
)


class ContextDef:
    __slots__ = ("_rng_counts", "_flags")

    def __init__(
        self,
        rng_counts: tp.Tuple[tp.Tuple[str, Counts], ...],
        flags: tp.Tuple[tp.Tuple[str, bool], ...],
    ):
        self._rng_counts = rng_counts
        self._flags = flags

    def merge(self, keys: tp.Mapping[str, KeyArray]) -> "Context":
        rngs = {
            name: RngStream(keys[name], count=0, count_path=count_path)
            for name, count_path in self._rng_counts
        }
        return Context(rngs=rngs, flags=dict(self._flags))


class PureContext(tp.Tuple[tp.Dict[str, KeyArray], ContextDef]):
    @classmethod
    def new(cls, keys: tp.Dict[str, KeyArray], contextdef: ContextDef):
        return cls((keys, contextdef))

    @property
    def keys(self) -> tp.Dict[str, KeyArray]:
        return self[0]

    @property
    def contextdef(self) -> ContextDef:
        return self[1]

    def merge(self):
        return self.contextdef.merge(self.keys)


def _pure_context_flatten(pure_context: PureContext):
    return tuple(pure_context), None


def _pure_context_unflatten(
    aux_data: None,
    children: tp.Tuple[tp.Dict[str, RngStream], ContextDef],
) -> PureContext:
    return PureContext(children)


jtu.register_pytree_node(PureContext, _pure_context_flatten, _pure_context_unflatten)


class Context:
    __slots__ = ("_rngs", "_flags")

    def __init__(
        self,
        rngs: tp.Mapping[str, RngStream],
        flags: tp.Mapping[str, bool],
    ):
        self._rngs = rngs
        self._flags = flags

    def has_rng(self, name: str) -> bool:
        return name in self._rngs

    def make_rng(self, name: str) -> KeyArray:
        if name not in self._rngs:
            raise ValueError(f"Unknown Rng Stream: {name}")
        return self._rngs[name].make_rng()

    def copy(self) -> "Context":
        return Context(rngs=self._rngs, flags=self._flags)

    def has_flag(self, name: str) -> bool:
        return name in self._flags

    def get_flag(self, name: str) -> tp.Optional[bool]:
        return self._flags.get(name, None)

    def partition(self) -> PureContext:
        rngs = {name: stream.fork() for name, stream in self._rngs.items()}
        keys = {name: stream._key for name, stream in rngs.items()}
        rng_counts = tuple((name, stream.count_path) for name, stream in rngs.items())
        return PureContext.new(keys, ContextDef(rng_counts, tuple(self._flags.items())))


def context(
    params: tp.Union[int, KeyArray, RngStream, None] = None,
    *,
    flags: tp.Optional[tp.Mapping[str, bool]] = None,
    **rngs: tp.Union[int, KeyArray, RngStream],
) -> Context:
    _flags = flags or {}

    if params is not None:
        rngs["params"] = params

    _rngs = {
        name: RngStream(jax.random.PRNGKey(value))
        if isinstance(value, int)
        else RngStream(value)
        if isinstance(value, jax.Array)
        else value
        for name, value in rngs.items()
    }

    return Context(rngs=_rngs, flags=_flags)
