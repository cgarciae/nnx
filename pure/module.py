from dataclasses import dataclass
from typing import Any
import jax
import jax.tree_util as jtu
import typing as tp
from pure.partitioning import Partition
from pure.state import State, Variable
from pure.rngs import Rngs, KeyArray

A = tp.TypeVar("A", contravariant=True)


class InitFn(tp.Protocol, tp.Generic[A]):
    def __call__(self, __key_or_stream: A, /, *args) -> tp.Any:
        ...


class Initializer:
    @tp.overload
    def __init__(
        self,
        initializer: InitFn[KeyArray],
        *args,
        collection: str = "params",
    ):
        ...

    @tp.overload
    def __init__(
        self,
        initializer: InitFn[Rngs],
        *args,
        stream: None,
        collection: str = "params",
    ):
        ...

    def __init__(
        self,
        initializer: tp.Union[InitFn[KeyArray], InitFn[Rngs]],
        *args,
        stream: tp.Optional[str] = "params",
        collection: str = "params",
    ):
        ...

    def create_variable(self, rngs: Rngs) -> Variable:
        ...


class Module:
    def create_state(self, rngs: Rngs) -> State:
        return State(
            (
                name,
                v.create_state(rngs)
                if isinstance(v, Module)
                else v.create_variable(rngs),
            )
            for name, v in vars(self).items()
            if isinstance(v, (Initializer, Module))
        )