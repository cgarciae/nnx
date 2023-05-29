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


class Partition(tp.Mapping[tp.Tuple[str, ...], Leaf]):
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
        return f"Partition({self._mapping})"


def _partition_flatten_with_keys(
    x: Partition,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[jtu.DictKey, Leaf], ...], tp.Tuple[tp.Tuple[str, ...], ...]
]:
    key_values = sorted(x.items(), key=lambda x: x[0])
    children = tuple((jtu.DictKey(key), value) for key, value in key_values)
    return children, tuple(key for key, _ in key_values)


def _partition_unflatten(keys: tp.Tuple[Path, ...], leaves: tp.Tuple[Leaf, ...]):
    return Partition(dict(zip(keys, leaves)))


jax.tree_util.register_pytree_with_keys(
    Partition, _partition_flatten_with_keys, _partition_unflatten
)


class Referential(tp.Generic[A], ABC):
    __slots__ = ("_collection", "_sharding")

    def __init__(self, collection: str, sharding: tp.Optional[Sharding]):
        self._collection = collection
        self._sharding = sharding

    @property
    @abstractmethod
    def value(self) -> A:
        ...

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def sharding(self) -> tp.Optional[Sharding]:
        return self._sharding


class Deref(Referential[A]):
    __slots__ = ()


@dataclasses.dataclass
class RefMetadata(tp.Generic[A]):
    value: A
    sharding: Sharding


def with_partitioning(
    initializer: initializers.Initializer,
    sharding: Sharding,
) -> initializers.Initializer:
    @functools.wraps(initializer)
    def wrapper(*args):
        return RefMetadata(initializer(*args), sharding)

    return wrapper  # type: ignore


def ref_metadata(value: A, sharding: Sharding) -> RefMetadata[A]:
    return RefMetadata(value, sharding)


class Ref(Referential[A]):
    __slots__ = ("_value", "_jax_trace", "_context_trace", "_trace_set")

    def __init__(
        self,
        value: tp.Union[A, RefMetadata[A]],
        collection: str,
        *,
        sharding: tp.Optional[Sharding] = None,
        context_trace: tp.Optional[tracers.MainTrace] = None,
    ):
        if isinstance(value, RefMetadata):
            if sharding is not None:
                raise ValueError(
                    "Cannot specify sharding when initializing from RefMetadata"
                )
            sharding = value.sharding
            value = value.value

        value = tp.cast(A, value)
        self._value = value
        self._jax_trace = tracers.current_jax_trace()
        self._context_trace = context_trace or self._jax_trace
        self._trace_set = frozenset((self._jax_trace, self._context_trace))
        super().__init__(collection, sharding)

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

    def to_index(self, val_path: tp.Tuple[str, ...]) -> "Index[A]":
        return Index(val_path, self._collection, self._sharding)


def ref(
    collection: str,
    value: tp.Union[A, RefMetadata[A]],
    sharding: tp.Optional[Sharding] = None,
    *,
    context_trace: tp.Optional[tracers.MainTrace] = None,
) -> A:
    return Ref(  # type: ignore
        value,
        collection=collection,
        sharding=sharding,
        context_trace=context_trace,
    )


def param(
    value: tp.Union[A, RefMetadata[A]],
    sharding: tp.Optional[Sharding] = None,
    *,
    context_trace: tp.Optional[tracers.MainTrace] = None,
) -> A:
    return ref(
        "params",
        value,
        sharding=sharding,
        context_trace=context_trace,
    )


class Value(Deref[A]):
    __slots__ = ("_value",)

    def __init__(
        self,
        value: A,
        collection: str,
        sharding: tp.Optional[Sharding],
    ):
        self._value = value
        super().__init__(collection, sharding)

    @property
    def value(self) -> A:
        return self._value

    def to_ref(self, context_trace: tracers.MainTrace) -> "Ref[A]":
        return Ref(
            self._value,
            collection=self.collection,
            sharding=self.sharding,
            context_trace=context_trace,
        )

    def __repr__(self) -> str:
        return (
            f"Value(collection={repr(self.collection)}, value={repr(self._value)}, "
            f"sharding={repr(self.sharding)})"
        )


def _value_flatten(
    x: Value[A],
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


class Index(Deref[A]):
    __slots__ = ("_val_path",)

    def __init__(
        self,
        val_path: tp.Tuple[str, ...],
        collection: str,
        sharding: tp.Optional[Sharding],
    ):
        self._val_path = val_path
        super().__init__(collection, sharding)

    @property
    def val_path(self) -> tp.Tuple[str, ...]:
        return self._val_path

    @property
    def value(self) -> A:
        raise ValueError(f"Cannot get value from '{type(self).__name__}' instances")

    def __repr__(self) -> str:
        return (
            f"Index(collection={self.collection}, val_path={self._val_path}, "
            f"sharding={self.sharding})"
        )


def _index_flatten(x: Index[A]):
    return (), (x._val_path, x._collection, x._sharding)


def _index_unflatten(
    metadata: tp.Tuple[tp.Tuple[str, ...], str, tp.Optional[Sharding]], _: tp.Tuple[()]
) -> Index[A]:
    return Index(*metadata)


jtu.register_pytree_node(Index, _index_flatten, _index_unflatten)
