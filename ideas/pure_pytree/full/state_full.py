import typing as tp
from types import MappingProxyType

import jax
import jax.tree_util as jtu
from pure.partitioning import Partition, StateDef, Variable

Node = tp.Union[Variable, "State"]
S = tp.TypeVar("S", bound="State")


class State(tp.Mapping[str, Node]):
    __slots__ = ("_variables",)

    def __init__(self, *args, **kwargs: tp.Union[Node, jax.Array]):
        self._variables = {
            k: self._create_node_field(v) for k, v in dict(*args, **kwargs).items()
        }

    @staticmethod
    def _create_node_field(value: tp.Any) -> Node:
        if isinstance(value, State):
            return value
        else:
            return Variable.from_value(value)

    @staticmethod
    def _update_node_field(node: Node, value: tp.Any) -> Node:
        if isinstance(node, State) and isinstance(value, State):
            return value
        elif isinstance(node, Variable) and isinstance(value, Variable):
            return node.update(value)
        else:
            raise ValueError(
                f"Cannot update node of type {type(node).__name__} with "
                f"value of type {type(value).__name__}"
            )

    def __getitem__(self, name: str) -> tp.Any:
        return self._variables[name].value

    __getattr__ = __getitem__

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    def keys(self) -> tp.KeysView[str]:
        return self._variables.keys()

    def values(self) -> tp.ValuesView[Node]:
        return self._variables.values()

    def __repr__(self) -> str:
        return f"State({self._variables})"

    def update(self, *args, **kwargs: tp.Union[Node, tp.Any]) -> "State":
        raise NotImplementedError

    @tp.overload
    def partition(self) -> tp.Tuple[tp.Dict[str, Partition], StateDef["State"]]:
        ...

    @tp.overload
    def partition(self, collection: str) -> tp.Tuple[Partition, StateDef["State"]]:
        ...

    @tp.overload
    def partition(
        self, collection: str, *collections: str
    ) -> tp.Tuple[tp.Tuple[Partition, ...], StateDef["State"]]:
        ...

    def partition(
        self, *collections: str
    ) -> tp.Tuple[
        tp.Union[tp.Dict[str, Partition], tp.Tuple[Partition, ...], Partition],
        StateDef["State"],
    ]:
        raise NotImplementedError

    @tp.overload
    def get_partition(self, collection: str) -> Partition:
        ...

    @tp.overload
    def get_partition(
        self, collection: str, *collections: str
    ) -> tp.Tuple[Partition, ...]:
        ...

    def get_partition(
        self, *collections: str
    ) -> tp.Union[Partition, tp.Tuple[Partition, ...]]:
        raise NotImplementedError

    def update_partition(self, partition: Partition, *partitions: Partition) -> "State":
        raise NotImplementedError

    @tp.overload
    def pop(self, name: str) -> Node:
        ...

    @tp.overload
    def pop(self, name: str, *names: str) -> tp.Tuple[Node, ...]:
        ...

    def pop(self, *names: str) -> tp.Union[Node, tp.Tuple[Node, ...]]:
        if len(names) == 0:
            raise ValueError("pop expected at least 1 argument, got 0")
        elif len(names) == 1:
            name = names[0]
            return self._variables.pop(name)
        else:
            return tuple(self._variables.pop(name) for name in names)


def _state_flatten_with_keys(state: State):
    nodes = tuple((jtu.GetAttrKey(name), variable) for name, variable in state.items())
    names = tuple(state)
    return nodes, names


def _state_unflatten(names: tp.Tuple[str, ...], nodes: tp.Tuple[Variable, ...]):
    return State(zip(names, nodes))


def _state_flatten(state: State):
    return tuple(state.values()), tuple(state)


jtu.register_pytree_with_keys(
    State, _state_flatten_with_keys, _state_unflatten, flatten_func=_state_flatten
)
