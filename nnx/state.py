import dataclasses
from functools import partial
import functools
from nnx import reprlib
import typing as tp

import jax
import jax.tree_util as jtu

from nnx import partitioning, tracers
from nnx.nn import initializers
from nnx.nodes import register_node_type
from nnx.reprlib import Config, Elem

A = tp.TypeVar("A")

Leaf = tp.Any
Path = tp.Tuple[str, ...]
Sharding = jax.sharding.PartitionSpec
StateDict = tp.Dict[Path, tp.Any]
StateMapping = tp.Mapping[Path, tp.Any]


class State(tp.Mapping[tp.Tuple[str, ...], Leaf], reprlib.Representable):
    __slots__ = ("_mapping",)

    def __init__(
        self,
        __input: tp.Union[
            tp.Mapping[tp.Tuple[str, ...], Leaf],
            tp.Iterator[tp.Tuple[tp.Tuple[str, ...], Leaf]],
        ],
        /,
    ):
        if isinstance(__input, tp.Mapping):
            self._mapping = dict(sorted(__input.items(), key=lambda x: x[0]))
        else:
            self._mapping = dict(sorted(__input, key=lambda x: x[0]))

    def get_collections(self) -> tp.Set[tp.Union[str, None]]:
        return {
            value.collection if isinstance(value, Variable) else None
            for value in self._mapping.values()
        }

    def __getitem__(self, __key: tp.Tuple[str, ...]) -> Leaf:
        return self._mapping[__key]

    def __iter__(self) -> tp.Iterator[tp.Tuple[str, ...]]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __nnx_repr__(self) -> tp.Iterator[tp.Union[Config, Elem]]:
        yield Config(type(self), value_sep=": ", parens_left="({", parens_right="})")

        for k, v in self._mapping.items():
            yield reprlib.Elem(str(k), v)

    @tp.overload
    def partition(self, first: partitioning.CollectionFilter, /) -> "State":
        ...

    @tp.overload
    def partition(
        self,
        first: partitioning.CollectionFilter,
        second: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple["State", ...]:
        ...

    def partition(
        self,
        first: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Union["State", tp.Tuple["State", ...]]:
        *states, rest = _split_state(self, first, *filters)

        if rest:
            raise ValueError(
                f"Non-exhaustive filters, got a non-empty remainder: "
                f"{list(rest.keys())}.\nUse `...` to match all remaining elements."
            )

        if len(states) == 1:
            states = State(states[0])
        else:
            states = tuple(State(state) for state in states)
        return states

    @tp.overload
    def filter(
        self,
        first: partitioning.CollectionFilter,
        /,
    ) -> "State":
        ...

    @tp.overload
    def filter(
        self,
        first: partitioning.CollectionFilter,
        second: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple["State", ...]:
        ...

    def filter(
        self,
        first: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Union["State", tp.Tuple["State", ...]]:
        (*states, _rest) = _split_state(self, first, *filters)

        assert len(states) == len(filters) + 1

        if len(states) == 1:
            states = State(states[0])
        else:
            states = tuple(State(state) for state in states)

        return states

    @staticmethod
    def merge(state: "State", /, *states: "State") -> "State":
        states = (state, *states)

        if len(states) == 1:
            return states[0]

        new_state: StateDict = {}

        for state in states:
            new_state.update(state)

        return State(new_state)

    def __or__(self, other: "State") -> "State":
        if not other:
            return self
        return State.merge(self, other)

    def __sub__(self, other: "State") -> "State":
        if not other:
            return self

        # create new State via __new__ to avoid __init__ sorting
        _mapping = {k: v for k, v in self.items() if k not in other}
        state = object.__new__(State)
        state._mapping = _mapping
        return state


def _state_flatten_with_keys(
    x: State,
):
    children = tuple((jtu.DictKey(key), value) for key, value in x.items())
    return children, tuple(x.keys())


def _state_unflatten(
    keys: tp.Tuple[Path, ...],
    leaves: tp.Tuple[Leaf, ...],
):
    state = object.__new__(State)
    state._mapping = dict(zip(keys, leaves))
    return state


jax.tree_util.register_pytree_with_keys(
    State, _state_flatten_with_keys, _state_unflatten
)


def _split_state(
    state: StateMapping,
    *filters: partitioning.CollectionFilter,
) -> tp.Tuple[StateDict, ...]:
    predicates = tuple(map(partitioning.to_predicate, filters))

    # we have n + 1 states, where n is the number of predicates
    # the last state is for values that don't match any predicate
    states: tp.Tuple[StateDict, ...] = tuple({} for _ in range(len(predicates) + 1))

    for path, value in state.items():
        for i, predicate in enumerate(predicates):
            if predicate(path, value):
                states[i][path] = value
                break
        else:
            # if we didn't break, set leaf to last state
            states[-1][path] = value

    return states


@dataclasses.dataclass
class VarMetadata(tp.Generic[A]):
    value: A
    sharding: Sharding


def with_partitioning(
    initializer: initializers.Initializer,
    sharding: Sharding,
) -> initializers.Initializer:
    @functools.wraps(initializer)
    def wrapper(*args):
        return VarMetadata(initializer(*args), sharding)

    return wrapper  # type: ignore


def var_metadata(value: A, sharding: Sharding) -> VarMetadata[A]:
    return VarMetadata(value, sharding)


class Variable(tp.Generic[A], reprlib.Representable):
    __slots__ = ("_value", "_collection", "_sharding")

    def __init__(
        self,
        value: A,
        collection: str,
        sharding: tp.Optional[Sharding],
    ):
        self._value = value
        self._collection = collection
        self._sharding = sharding

    @property
    def value(self) -> A:
        return self._value

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def sharding(self) -> tp.Optional[Sharding]:
        return self._sharding

    def __nnx_repr__(self):
        yield reprlib.Config(type=type(self))
        yield reprlib.Elem("collection", self._collection)
        yield reprlib.Elem("value", self._value)
        if self._sharding is not None:
            yield reprlib.Elem("sharding", self._sharding)

    def copy(self) -> "Variable[A]":
        return Variable(self._value, self._collection, self._sharding)

    def replace(
        self,
        **kwargs: tp.Any,
    ) -> "Variable[A]":
        updates: tp.Dict[str, tp.Any] = {
            "value": self._value,
            "collection": self._collection,
            "sharding": self._sharding,
        }
        updates.update(kwargs)
        return Variable(**updates)


def _variable_flatten(
    x: Variable[tp.Any],
    *,
    with_keys: bool,
):
    if with_keys:
        node = (jtu.GetAttrKey("value"), x._value)
    else:
        node = x._value

    return (node,), (x._collection, x._sharding)


def _variable_unflatten(
    metadata: tp.Tuple[str, tp.Optional[Sharding]], children: tp.Tuple[A]
) -> Variable[A]:
    return Variable(children[0], *metadata)


jtu.register_pytree_with_keys(
    Variable,
    partial(_variable_flatten, with_keys=True),
    _variable_unflatten,
    flatten_func=partial(_variable_flatten, with_keys=False),
)


def var(
    collection: str,
    value: tp.Union[A, VarMetadata[A]],
    sharding: tp.Optional[Sharding] = None,
) -> A:
    return Variable(  # type: ignore
        value,
        collection=collection,
        sharding=sharding,
    )


def param(
    value: tp.Union[A, VarMetadata[A]],
    sharding: tp.Optional[Sharding] = None,
) -> A:
    return var("params", value, sharding=sharding)


# register nodes
register_node_type(State)
register_node_type(Variable)
