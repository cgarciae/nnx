from abc import ABC, abstractmethod
import dataclasses
from functools import partial
import functools
import typing as tp

import jax
import jax.tree_util as jtu

from nnx import tracers
from nnx.nn import initializers

A = tp.TypeVar("A")

Leaf = tp.Any
Path = tp.Tuple[str, ...]
Sharding = jax.sharding.PartitionSpec


class State(tp.Mapping[tp.Tuple[str, ...], Leaf]):
    def __init__(
        self,
        __input: tp.Union[
            tp.Mapping[tp.Tuple[str, ...], Leaf],
            tp.Iterator[tp.Tuple[tp.Tuple[str, ...], Leaf]],
        ],
        /,
    ):
        self._mapping = dict(__input)

    def __getitem__(self, __key: tp.Tuple[str, ...]) -> Leaf:
        return self._mapping[__key]

    def __iter__(self) -> tp.Iterator[tp.Tuple[str, ...]]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"State({self._mapping})"


def _partition_flatten_with_keys(
    x: State,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[jtu.DictKey, Leaf], ...], tp.Tuple[tp.Tuple[str, ...], ...]
]:
    key_values = sorted(x.items(), key=lambda x: x[0])
    children = tuple((jtu.DictKey(key), value) for key, value in key_values)
    return children, tuple(key for key, _ in key_values)


def _partition_unflatten(keys: tp.Tuple[Path, ...], leaves: tp.Tuple[Leaf, ...]):
    return State(dict(zip(keys, leaves)))


jax.tree_util.register_pytree_with_keys(
    State, _partition_flatten_with_keys, _partition_unflatten
)


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


class Variable(tp.Generic[A]):
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

        value = tp.cast(A, value)
        self._value = value
        self._collection = collection
        self._sharding = sharding
        self._jax_trace = tracers.current_jax_trace()
        self._context_trace = context_trace or self._jax_trace
        self._trace_set = frozenset((self._jax_trace, self._context_trace))

    def __repr__(self) -> str:
        return (
            f"Variable(value={self._value}, collection={self._collection}, "
            f"sharding={self._sharding})"
        )

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

    def to_value(self) -> "Value[A]":
        return Value(self._value, self._collection, self._sharding)

    def copy(self) -> "Variable[A]":
        ref = object.__new__(Variable)
        ref._value = self._value
        ref._jax_trace = self._jax_trace
        ref._context_trace = self._context_trace
        ref._trace_set = self._trace_set
        ref._collection = self._collection
        ref._sharding = self._sharding
        return ref


class Value(tp.Generic[A]):
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

    def __repr__(self) -> str:
        return (
            f"Value(value={self._value}, collection={self._collection}, "
            f"sharding={self._sharding})"
        )

    def to_var(self, context_trace: tracers.MainTrace) -> Variable[A]:
        return Variable(
            self._value,
            self._collection,
            sharding=self._sharding,
            context_trace=context_trace,
        )


def _value_flatten(
    x: Value[tp.Any],
    *,
    with_keys: bool,
):
    if with_keys:
        node = (jtu.GetAttrKey("value"), x._value)
    else:
        node = x._value

    return (node,), (x._collection, x._sharding)


def _value_unflatten(
    metadata: tp.Tuple[str, tp.Optional[Sharding]], children: tp.Tuple[A]
) -> Value[A]:
    return Value(children[0], *metadata)


jtu.register_pytree_with_keys(
    Value,
    partial(_value_flatten, with_keys=True),
    _value_unflatten,
    flatten_func=partial(_value_flatten, with_keys=False),
)


def var(
    collection: str,
    value: tp.Union[A, VarMetadata[A]],
    sharding: tp.Optional[Sharding] = None,
    *,
    context_trace: tp.Optional[tracers.MainTrace] = None,
) -> A:
    return Variable(  # type: ignore
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
