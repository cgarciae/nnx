from types import MappingProxyType
import typing as tp
import jax
import hashlib

from nnx import tracers
import jax.tree_util as jtu

KeyArray = tp.Union[jax.Array, jax.random.KeyArray]


def _stable_hash(data: tp.Tuple[int, ...]) -> int:
    hash_str = " ".join(str(x) for x in data)
    _hash = hashlib.blake2s(hash_str.encode())
    hash_bytes = _hash.digest()
    # uint32 is represented as 4 bytes in big endian
    return int.from_bytes(hash_bytes[:4], byteorder="big")


class RngStream:
    __slots__ = (
        "_key",
        "_count",
        "_count_path",
        "_jax_trace",
        "_context_trace",
    )

    def __init__(
        self,
        key: KeyArray,
        count: int = 0,
        count_path: tp.Tuple[int, ...] = (),
        *,
        context_trace: tp.Optional[tracers.MainTrace] = None,
    ):
        self._key = key
        self._count = count
        self._count_path = count_path
        self._jax_trace = tracers.current_jax_trace()
        self._context_trace = context_trace or self._jax_trace

    def _validate_trace(self):
        value_trace = tracers.get_top_trace(self._key)
        if self._jax_trace is not tracers.current_jax_trace() or (
            value_trace is not self._jax_trace
            and value_trace is not self._context_trace
        ):
            raise ValueError("Rng used in a different trace")

    @property
    def key(self) -> jax.random.KeyArray:
        self._validate_trace()
        return self._key

    @property
    def count(self) -> int:
        return self._count

    @property
    def count_path(self) -> tp.Tuple[int, ...]:
        return self._count_path

    def next(self) -> jax.random.KeyArray:
        self._validate_trace()
        fold_data = _stable_hash(self._count_path + (self._count,))
        self._count += 1
        return jax.random.fold_in(self._key, fold_data)

    def fork(self) -> "RngStream":
        self._validate_trace()
        count_path = self._count_path + (self._count,)
        self._count += 1
        return RngStream(self._key, count_path=count_path)


def _rng_stream_flatten_with_keys(
    rng: RngStream,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[tp.Hashable, jax.random.KeyArray], ...],
    tp.Tuple[int, tp.Tuple[int, ...]],
]:
    return ((jtu.GetAttrKey("key"), rng.key),), (rng.count, rng.count_path)


def _rng_stream_unflatten(
    aux_data: tp.Tuple[int, tp.Tuple[int, ...]],
    children: tp.Tuple[jax.random.KeyArray, ...],
) -> RngStream:
    count, count_path = aux_data
    key = children[0]
    return RngStream(key, count, count_path)


def _rng_stream_flatten(rng: RngStream):
    return (rng.key,), (rng.count, rng.count_path)


jax.tree_util.register_pytree_with_keys(
    RngStream,
    _rng_stream_flatten_with_keys,
    _rng_stream_unflatten,
    flatten_func=_rng_stream_flatten,
)


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
        context_trace = tracers.get_top_trace(rngs)

        if rngs is None:
            _rngs = {}
        elif isinstance(rngs, tp.Mapping):
            _rngs = dict(rngs)
        elif isinstance(rngs, RngStream):
            _rngs = dict(params=rngs)
        else:
            _rngs = dict(params=RngStream(rngs, context_trace=context_trace))

        _rngs.update(**rng_updates)

        _rngs = {
            name: RngStream(key, context_trace=context_trace)
            if not isinstance(key, RngStream)
            else key
            for name, key in _rngs.items()
        }

        self._rngs = _rngs
        self._flags = MappingProxyType(flags or {})

    @property
    def rngs(self) -> tp.Mapping[str, RngStream]:
        return MappingProxyType(self._rngs)

    @property
    def flags(self) -> tp.Mapping[str, bool]:
        return self._flags

    def has_rng(self, name: str) -> bool:
        return name in self._rngs

    def make_rng(self, name: str) -> KeyArray:
        if name not in self._rngs:
            raise ValueError(f"Unknown Rng Stream: {name}")
        return self._rngs[name].next()

    def fork(self) -> "Context":
        return Context(
            rngs={name: stream.fork() for name, stream in self._rngs.items()},
            flags=self._flags,
        )

    def copy(self) -> "Context":
        return Context(rngs=self._rngs, flags=self._flags)

    def has_flag(self, name: str) -> bool:
        return name in self._flags

    def get_flag(self, name: str) -> tp.Optional[bool]:
        return self._flags.get(name, None)


def _context_flatten_with_keys(ctx: Context):
    node = (jtu.GetAttrKey("rngs"), ctx._rngs)
    return (node,), tuple(ctx._flags.items())


def _context_unflatten(
    metadata: tp.Tuple[tp.Tuple[str, bool], ...],
    nodes: tp.Tuple[tp.Dict[str, RngStream]],
) -> Context:
    return Context(rngs=nodes[0], flags=dict(metadata))


def _context_flatten(ctx: Context):
    node = ctx._rngs
    return (node,), tuple(ctx._flags.items())


jtu.register_pytree_with_keys(
    Context,
    _context_flatten_with_keys,
    _context_unflatten,
    flatten_func=_context_flatten,
)
