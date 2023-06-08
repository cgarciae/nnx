import hashlib
import typing as tp
from types import MappingProxyType

import jax
import jax.tree_util as jtu

from nnx import errors, tracers

KeyArray = tp.Union[jax.Array, jax.random.KeyArray]


def _stable_hash(data: tp.Tuple[int, ...]) -> int:
    hash_str = " ".join(str(x) for x in data)
    _hash = hashlib.blake2s(hash_str.encode())
    hash_bytes = _hash.digest()
    # uint32 is represented as 4 bytes in big endian
    return int.from_bytes(hash_bytes[:4], byteorder="big")


class RngStream:
    __slots__ = ("_key", "_count", "_count_path", "_trace_state")

    def __init__(
        self, key: KeyArray, count: int = 0, count_path: tp.Tuple[int, ...] = ()
    ):
        self._key = key
        self._count = count
        self._count_path = count_path
        self._trace_state = tracers.TraceState()

    @property
    def count(self) -> int:
        return self._count

    @property
    def count_path(self) -> tp.Tuple[int, ...]:
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
    tp.Tuple[int, tp.Tuple[int, ...]],
]:
    return ((jtu.GetAttrKey("key"), rng._key),), (rng.count, rng.count_path)


def _rng_stream_unflatten(
    aux_data: tp.Tuple[int, tp.Tuple[int, ...]],
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
    __slots__ = ("_flags",)

    def __init__(self, flags: tp.Tuple[tp.Tuple[str, bool], ...]):
        self._flags = flags

    def merge(self, **rngs: RngStream) -> "Context":
        return Context(rngs=rngs, flags=dict(self._flags))


class PureContext(tp.Tuple[tp.Dict[str, RngStream], ContextDef]):
    @classmethod
    def new(cls, rngs: tp.Dict[str, RngStream], contextdef: ContextDef):
        return cls((rngs, contextdef))

    @property
    def rngs(self) -> tp.Dict[str, RngStream]:
        return self[0]

    @property
    def contextdef(self) -> ContextDef:
        return self[1]

    def merge(self):
        return self.contextdef.merge(self.rngs)


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
        rngs: tp.Union[
            tp.Mapping[str, tp.Union[RngStream, KeyArray]], RngStream, KeyArray, None
        ] = None,
        *,
        flags: tp.Optional[tp.Mapping[str, bool]] = None,
        **rng_updates: tp.Union[RngStream, KeyArray],
    ):
        if rngs is None:
            _rngs = {}
        elif isinstance(rngs, tp.Mapping):
            _rngs = dict(rngs)
        elif isinstance(rngs, RngStream):
            _rngs = dict(params=rngs)
        else:
            _rngs = dict(params=RngStream(rngs))

        _rngs.update(**rng_updates)

        _rngs = {
            name: RngStream(key) if not isinstance(key, RngStream) else key
            for name, key in _rngs.items()
        }

        self._rngs = _rngs
        self._flags = MappingProxyType(flags or {})

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
        return PureContext.new(rngs, ContextDef(tuple(self._flags.items())))
