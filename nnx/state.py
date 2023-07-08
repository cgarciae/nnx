import typing as tp

import jax
import jax.tree_util as jtu

from nnx import nodes, partitioning, reprlib
from nnx.containers import Node

A = tp.TypeVar("A")

Leaf = tp.Any
Path = str
StateDict = tp.Dict[Path, tp.Any]
StateMapping = tp.Mapping[Path, tp.Any]


class State(tp.Mapping[Path, Leaf], reprlib.Representable):
    __slots__ = ("_mapping",)

    def __init__(
        self,
        __input: tp.Union[
            tp.Mapping[Path, Leaf],
            tp.Iterator[tp.Tuple[Path, Leaf]],
        ],
        /,
    ):
        if isinstance(__input, tp.Mapping):
            self._mapping = dict(sorted(__input.items(), key=lambda x: x[0]))
        else:
            self._mapping = dict(sorted(__input, key=lambda x: x[0]))

    def __getitem__(self, __key: Path) -> Leaf:
        return self._mapping[__key]

    def __iter__(self) -> tp.Iterator[Path]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __nnx_repr__(self):
        yield reprlib.Object(type(self), value_sep=": ", start="({", end="})")

        for k, v in self._mapping.items():
            yield reprlib.Attr(repr(k), v)

    @tp.overload
    def partition(self, first: partitioning.Filter, /) -> "State":
        ...

    @tp.overload
    def partition(
        self,
        first: partitioning.Filter,
        second: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> tp.Tuple["State", ...]:
        ...

    def partition(
        self, first: partitioning.Filter, /, *filters: partitioning.Filter
    ) -> tp.Union["State", tp.Tuple["State", ...]]:
        filters = (first, *filters)
        *states, rest = _split_state(self, *filters)

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
        first: partitioning.Filter,
        /,
    ) -> "State":
        ...

    @tp.overload
    def filter(
        self,
        first: partitioning.Filter,
        second: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> tp.Tuple["State", ...]:
        ...

    def filter(
        self,
        first: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> tp.Union["State", tp.Tuple["State", ...]]:
        *states, _rest = _split_state(self, first, *filters)

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
    *filters: partitioning.Filter,
) -> tp.Tuple[StateDict, ...]:
    for i, filter_ in enumerate(filters):
        if filter_ is ... and i != len(filters) - 1:
            raise ValueError(
                f"Ellipsis `...` can only be used as the last filter, "
                f"got it at index {i}."
            )
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


# register nodes
nodes.register_node_type(State)
