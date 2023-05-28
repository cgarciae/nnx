from abc import ABC, abstractmethod
from functools import partial
from types import MappingProxyType
import typing as tp

import jax
import jax.tree_util as jtu

from nnx import tracers

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
        if isinstance(__input, tp.Mapping):
            self._mapping = MappingProxyType(dict(__input))
        else:
            self._mapping = MappingProxyType(dict(__input))

    def __getitem__(self, __key: tp.Tuple[str, ...]) -> Leaf:
        return self._mapping[__key]

    def __iter__(self) -> tp.Iterator[tp.Tuple[str, ...]]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)


def _partition_flatten_with_keys(
    x: Partition,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[jtu.DictKey, Leaf], ...], tp.Tuple[tp.Tuple[str, ...], ...]
]:
    children = tuple((jtu.DictKey(key), value) for key, value in x.items())
    return children, tuple(x.keys())


def _partition_unflatten(keys: tp.Tuple[Path, ...], leaves: tp.Tuple[Leaf, ...]):
    return Partition(dict(zip(keys, leaves)))


jax.tree_util.register_pytree_with_keys(
    Partition, _partition_flatten_with_keys, _partition_unflatten
)


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"  # pragma: no cover


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jtu.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


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


class Ref(Referential[A]):
    __slots__ = ("_value", "_jax_trace", "_context_trace", "_trace_set")

    def __init__(
        self,
        value: A,
        *,
        collection: str = "",
        sharding: tp.Optional[Sharding] = None,
        context_trace: tp.Optional[tracers.MainTrace] = None,
    ):
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

    def to_ref(self) -> "Ref[A]":
        return Ref(self._value, collection=self.collection, sharding=self.sharding)

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
