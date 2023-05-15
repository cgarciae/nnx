from types import MappingProxyType
import typing as tp

import jax
import jax.tree_util as jtu

from pure.partitioning import StateDef, Variable, Partition


Node = tp.Union[Variable, "State"]
S = tp.TypeVar("S", bound="State")


class State(tp.Mapping[str, Node]):
    __slots__ = ("_variables",)

    def __init__(self, *args, **kwargs: tp.Union[Node, jax.Array]):
        ...

    def __getitem__(self, name: str) -> tp.Any:
        ...

    __getattr__ = __getitem__

    def __iter__(self) -> tp.Iterator[str]:
        ...

    def __len__(self) -> int:
        ...

    def keys(self) -> tp.KeysView[str]:
        ...

    def values(self) -> tp.ValuesView[Node]:
        ...

    def __repr__(self) -> str:
        ...

    def update(self, *args, **kwargs: tp.Union[Node, tp.Any]) -> "State":
        ...

    @tp.overload
    def partition(self) -> tp.Dict[str, Partition]:
        ...

    @tp.overload
    def partition(self, collection: str) -> Partition:
        ...

    @tp.overload
    def partition(self, collection: str, *collections: str) -> tp.Tuple[Partition, ...]:
        ...

    def partition(
        self, *collections: str
    ) -> tp.Union[tp.Dict[str, Partition], tp.Tuple[Partition, ...], Partition]:
        ...

    def merge(self, partition: Partition, *partitions: Partition) -> "State":
        ...

    @tp.overload
    def pop(self, name: str) -> Node:
        ...

    @tp.overload
    def pop(self, name: str, *names: str) -> tp.Tuple[Node, ...]:
        ...

    def pop(self, *names: str) -> tp.Union[Node, tp.Tuple[Node, ...]]:
        ...


def _state_flatten_with_keys(state: State):
    ...


def _state_unflatten(names: tp.Tuple[str, ...], nodes: tp.Tuple[Variable, ...]):
    ...


def _state_flatten(state: State):
    ...


jtu.register_pytree_with_keys(
    State, _state_flatten_with_keys, _state_unflatten, flatten_func=_state_flatten
)


def merge(partition: Partition, other: Partition, *rest: Partition) -> State:
    ...
