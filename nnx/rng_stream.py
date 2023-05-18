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


class Rngs(tp.Mapping[str, RngStream]):
    __slots__ = ("_streams",)

    def __init__(
        self,
        streams: tp.Optional[tp.Mapping[str, tp.Union[RngStream, KeyArray]]] = None,
        **kwargs: tp.Union[RngStream, KeyArray],
    ):
        if streams is None:
            streams = {}
        else:
            streams = dict(streams)

        streams.update(kwargs)

        streams = {
            name: RngStream(key) if isinstance(key, jax.Array) else key
            for name, key in streams.items()
        }

        self._streams = MappingProxyType(streams)

    def __getitem__(self, key: str) -> RngStream:
        return self._streams[key]

    def __iter__(self):
        return iter(self._streams)

    def __len__(self):
        return len(self._streams)

    def __contains__(self, key):
        return key in self._streams

    def make_rng(self, name: str) -> KeyArray:
        if name not in self:
            raise ValueError(f"Unknown Rng Stream: {name}")
        return self._streams[name].next()

    def fork(self) -> "Rngs":
        return Rngs({name: stream.fork() for name, stream in self._streams.items()})

    def copy(self) -> "Rngs":
        return Rngs(self._streams.copy())


def _rngs_flatten_with_keys(rngs: Rngs):
    nodes = tuple((jtu.GetAttrKey(name), stream) for name, stream in rngs.items())
    names = tuple(rngs.keys())
    return nodes, names


def _rngs_unflatten(
    names: tp.Tuple[str, ...],
    nodes: tp.Tuple[RngStream, ...],
) -> Rngs:
    return Rngs({name: stream for name, stream in zip(names, nodes)})


def _rngs_flatten(rngs: Rngs):
    return tuple(rngs.values()), tuple(rngs.keys())


jtu.register_pytree_with_keys(
    Rngs, _rngs_flatten_with_keys, _rngs_unflatten, flatten_func=_rngs_flatten
)
