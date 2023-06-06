import typing as tp

import jax

KeyArray = tp.Union[jax.Array, jax.random.KeyArray]


class RngStream:
    def __init__(
        self, key: KeyArray, count: int = 0, count_path: tp.Tuple[int, ...] = ()
    ):
        ...

    @property
    def key(self) -> jax.random.KeyArray:
        ...

    @property
    def count(self) -> int:
        ...

    @property
    def count_path(self) -> tp.Tuple[int, ...]:
        ...

    def next(self) -> jax.random.KeyArray:
        ...

    def fork(self) -> "RngStream":
        ...


class Rngs:
    def __init__(self, **streams: tp.Union[KeyArray, RngStream]):
        ...

    def make_rng(self, stream: str) -> jax.Array:
        ...

    @tp.overload
    def fork(self, stream: str) -> RngStream:
        ...

    @tp.overload
    def fork(self, stream: str, *streams: str) -> tp.Tuple[RngStream, ...]:
        ...

    def fork(self, *streams: str) -> tp.Union[RngStream, tp.Tuple[RngStream, ...]]:
        ...
