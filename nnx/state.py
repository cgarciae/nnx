import dataclasses
from functools import partial
import functools
from nnx import reprlib
import typing as tp

import jax
import jax.tree_util as jtu

from nnx import partitioning, tracers
from nnx.nn import initializers
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
            yield reprlib.Elem(str(k), repr(v))

    @tp.overload
    def split(
        self,
        first: None = None,
        second: None = None,
        /,
    ) -> "State":
        ...

    @tp.overload
    def split(
        self, first: partitioning.CollectionFilter, second: None = None, /
    ) -> "State":
        ...

    @tp.overload
    def split(
        self,
        first: partitioning.CollectionFilter,
        second: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple["State", ...]:
        ...

    def split(
        self,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Union["State", tp.Tuple["State", ...]]:
        if len(filters) == 0 or (len(filters) == 1 and filters[0] is Ellipsis):
            return self
        else:
            (*states, rest) = _split_state(self, *filters)

            if len(rest) > 0:
                raise ValueError(
                    f"Non-exhaustive filters, got a non-empty remainder: {rest}.\n"
                    f"Use `...` to match all remaining elements."
                )

            if len(states) == 1:
                states = State(states[0])
            else:
                states = tuple(State(state) for state in states)

        return states

    @tp.overload
    def get(
        self,
        filter: partitioning.CollectionFilter,
        /,
    ) -> "State":
        ...

    @tp.overload
    def get(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple["State", ...]:
        ...

    def get(
        self, *filters: partitioning.CollectionFilter
    ) -> tp.Union["State", tp.Tuple["State", ...]]:
        if len(filters) == 0:
            raise ValueError("Expected at least one filter")

        (*states, _rest) = _split_state(self, *filters)

        assert len(states) == len(filters)

        if len(states) == 1:
            states = State(states[0])
        else:
            states = tuple(State(state) for state in states)

        return states

    @tp.overload
    @staticmethod
    def merge(state: "State", *states: "State") -> "State":
        ...

    @staticmethod
    def merge(*states: "State") -> "State":
        if len(states) == 0:
            raise ValueError("Expected at least one state")
        elif len(states) == 1:
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


class MutableVariable(tp.Generic[A], reprlib.Representable):
    __slots__ = (
        "_value",
        "_collection",
        "_sharding",
        "_jax_trace",
        "_context_trace",
        "_trace_set",
    )

    def __init__(
        self,
        value: tp.Union[A, VarMetadata[A]],
        collection: str,
        *,
        sharding: tp.Optional[Sharding] = None,
        context_trace: tp.Optional[tracers.MainTrace] = None,
    ):
        if isinstance(value, VarMetadata):
            if sharding is not None:
                raise ValueError(
                    "Cannot specify sharding when initializing from RefMetadata"
                )
            sharding = value.sharding
            value = value.value

        if collection == "rest":
            raise ValueError(
                "'rest' is a reserved collection name, please use a different name."
            )

        value = tp.cast(A, value)
        self._value = value
        self._collection = collection
        self._sharding = sharding
        self._jax_trace = tracers.current_jax_trace()
        self._context_trace = context_trace or self._jax_trace
        self._trace_set = frozenset((self._jax_trace, self._context_trace))

    def __nnx_repr__(self):
        yield reprlib.Config(type(self))
        yield reprlib.Elem("collection", repr(self._collection))
        yield reprlib.Elem("value", repr(self._value))
        if self._sharding is not None:
            yield reprlib.Elem("sharding", repr(self._sharding))

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def sharding(self) -> tp.Optional[Sharding]:
        return self._sharding

    @property
    def value(self) -> A:
        # TODO: passing references as a constant to a function as a capture should
        # be allowed? Commenting out for now.
        # value_trace = tracers.get_top_trace(self._value)
        # if self._jax_trace is not tracers.current_jax_trace() or (
        #     value_trace is not self._jax_trace
        #     and value_trace is not self._context_trace
        # ):
        #     raise ValueError("Cannot access ref from different trace level")
        return self._value

    @value.setter
    def value(self, value: A):
        value_trace = tracers.get_top_trace(self._value)
        if self._jax_trace is not tracers.current_jax_trace() or (
            value_trace is not self._jax_trace
            and value_trace is not self._context_trace
        ):
            raise ValueError("Cannot mutate ref from different trace level")

        invalid_traces = tracers.get_all_traces(value) - self._trace_set
        if invalid_traces:
            raise ValueError(
                "Cannot mutate ref with value that contains tracers from other "
                f"traces: {invalid_traces}"
            )

        self._value = value

    def to_immutable(self) -> "Variable[A]":
        return Variable(self._value, self._collection, self._sharding)

    def copy(self) -> "MutableVariable[A]":
        ref = object.__new__(MutableVariable)
        ref._value = self._value
        ref._jax_trace = self._jax_trace
        ref._context_trace = self._context_trace
        ref._trace_set = self._trace_set
        ref._collection = self._collection
        ref._sharding = self._sharding
        return ref


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
        yield reprlib.Config(type(self))
        yield reprlib.Elem("collection", repr(self._collection))
        yield reprlib.Elem("value", repr(self._value))
        yield reprlib.Elem("sharding", repr(self._sharding))

    def to_mutable(self, context_trace: tracers.MainTrace) -> MutableVariable[A]:
        return MutableVariable(
            self._value,
            self._collection,
            sharding=self._sharding,
            context_trace=context_trace,
        )


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
    *,
    context_trace: tp.Optional[tracers.MainTrace] = None,
) -> A:
    return MutableVariable(  # type: ignore
        value,
        collection=collection,
        sharding=sharding,
        context_trace=context_trace,
    )


def param(
    value: tp.Union[A, VarMetadata[A]],
    sharding: tp.Optional[Sharding] = None,
    *,
    context_trace: tp.Optional[tracers.MainTrace] = None,
) -> A:
    return var(
        "params",
        value,
        sharding=sharding,
        context_trace=context_trace,
    )
