import typing as tp
import jax
import hashlib

from refx import tracers
import jax.tree_util as jtu


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
        "_refx_trace",
    )

    def __init__(
        self,
        key: jax.random.KeyArray,
        count: int = 0,
        count_path: tp.Tuple[int, ...] = (),
    ):
        self._key = key
        self._count = count
        self._count_path = count_path
        self._jax_trace = tracers.current_jax_trace()
        self._refx_trace = tracers.current_refx_trace()

    def _validate_trace(self):
        if (
            self._jax_trace is not tracers.current_jax_trace()
            or self._refx_trace is not tracers.current_refx_trace()
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


def _rng_flatten_with_keys(
    rng: RngStream,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[tp.Hashable, jax.random.KeyArray], ...],
    tp.Tuple[int, tp.Tuple[int, ...]],
]:
    return ((jtu.GetAttrKey("key"), rng.key),), (rng.count, rng.count_path)


def _rng_unflatten(
    aux_data: tp.Tuple[int, tp.Tuple[int, ...]],
    children: tp.Tuple[jax.random.KeyArray, ...],
) -> RngStream:
    count, count_path = aux_data
    key = children[0]
    return RngStream(key, count, count_path)


jax.tree_util.register_pytree_with_keys(
    RngStream, _rng_flatten_with_keys, _rng_unflatten
)
