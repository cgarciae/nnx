import dataclasses
import functools
import typing as tp
from abc import ABC, ABCMeta, abstractmethod
from functools import partial
from types import MappingProxyType
from typing import Any

import jax.tree_util as jtu

from nnx import nodes, reprlib
from nnx.nn import initializers

A = tp.TypeVar("A")
B = tp.TypeVar("B")
C = tp.TypeVar("C", bound="Container[tp.Any]")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
Sharding = tp.Tuple[tp.Optional[str], ...]
PARTITION_NAME = "partition_name"


class Container(ABC, tp.Generic[A]):
    __slots__ = ("_value",)

    def __init__(self, value: A):
        self._value = value

    @property
    def value(self) -> A:
        return self._value

    def replace_value(self: C, value: tp.Any) -> C:
        if isinstance(value, Container):
            if not self.is_equivalent(value):
                raise ValueError(
                    "Cannot replace value from incompatible container, "
                    f"expected {self}, got {value}"
                )
            value = value.value

        return self._replace_value(value)

    @abstractmethod
    def _replace_value(self: "Container[A]", value: B) -> "Container[B]":
        ...

    @abstractmethod
    def is_equivalent(self, other: tp.Any) -> bool:
        ...

    @abstractmethod
    def copy(self: "Container[A]") -> "Container[A]":
        ...


@dataclasses.dataclass
class NodeMetadata(tp.Generic[A]):
    value: A
    metadata: tp.Mapping[str, tp.Any]


@dataclasses.dataclass(repr=False)
class MetadataRepr(reprlib.Representable):
    metadata: tp.Mapping[str, tp.Any]

    def __nnx_repr__(self):
        yield reprlib.Object(type="", start="{", end="}", value_sep=": ")
        for name, value in self.metadata.items():
            yield reprlib.Attr(repr(name), repr(value))


class Node(Container[A], reprlib.Representable):
    __slots__ = ("_value", "_metadata")

    def __init__(
        self,
        value: tp.Union[A, NodeMetadata[A]],
        **metadata: tp.Any,
    ):
        if isinstance(value, NodeMetadata):
            metadata = {**value.metadata, **metadata}
            value = tp.cast(A, value.value)

        self._value = value
        self._metadata = MappingProxyType(metadata)

    def __getattr__(self, name: str) -> tp.Any:
        if name in self._metadata:
            return self._metadata[name]
        raise AttributeError(
            f"Variable has no attribute '{name}' in metadata: {self._metadata}"
        )

    @property
    def metadata(self) -> tp.Mapping[str, tp.Any]:
        return self._metadata

    def add_axis(self, index: int, params: tp.Dict[tp.Any, tp.Any]):
        axis_name = self._get_partition_name(params)
        sharding = self.sharding
        assert isinstance(sharding, tuple)
        sharding = list(sharding)

        while len(sharding) < index:
            sharding.append(None)
        sharding.insert(index, axis_name)
        return self.replace_metadata(sharding=tuple(sharding))

    def remove_axis(
        self: "Node[A]", index: int, params: tp.Dict[tp.Any, tp.Any]
    ) -> "Node[A]":
        axis_name = self._get_partition_name(params)
        names = list(self.names)
        assert names.pop(index) == axis_name
        return self.replace(names=tuple(names))

    def _get_partition_name(self, params: tp.Dict[tp.Any, tp.Any]) -> str:
        if PARTITION_NAME not in params:
            raise ValueError(
                f'Trying to transform a Partitioned variable but "partition_name"'
                f" is not specified in metadata_params: {self}"
            )
        return params[PARTITION_NAME]

    def _replace_value(self: "Node[A]", value: B) -> "Node[B]":
        return Node(value, **self._metadata)

    def replace_metadata(self: "Node[A]", **kwargs) -> "Node[A]":
        metadata = dict(self.metadata)
        metadata.update(**kwargs)
        return Node(self._value, **metadata)

    def is_equivalent(self, other: tp.Any) -> bool:
        return type(other) is Node and self._metadata == other._metadata

    def copy(self: "Node[A]") -> "Node[A]":
        return Node(self._value, **self._metadata)

    def __eq__(self, other: object) -> bool:
        return type(other) is Node and self._metadata == other._metadata

    def __nnx_repr__(self):
        yield reprlib.Object(type=type(self))
        yield reprlib.Attr(
            "value", repr(self._value) if isinstance(self._value, str) else self._value
        )
        if self._metadata:
            yield reprlib.Attr("metadata", MetadataRepr(self._metadata))


def _node_flatten(
    x: Node[tp.Any],
    *,
    with_keys: bool,
):
    if with_keys:
        node = (jtu.GetAttrKey("value"), x._value)
    else:
        node = x._value

    return (node,), x._metadata


def _node_unflatten(
    metadata: tp.Mapping[str, tp.Any], children: tp.Tuple[A]
) -> Node[A]:
    return Node(children[0], **metadata)


jtu.register_pytree_with_keys(
    Node,
    partial(_node_flatten, with_keys=True),
    _node_unflatten,
    flatten_func=partial(_node_flatten, with_keys=False),
)


def with_partitioning(
    initializer: initializers.Initializer,
    sharding: Sharding,
) -> initializers.Initializer:
    @functools.wraps(initializer)
    def wrapper(*args):
        return NodeMetadata(initializer(*args), metadata={"sharding": sharding})

    return wrapper  # type: ignore


class CheckableMeta(ABCMeta):
    def __instancecheck__(cls, instance: Any) -> bool:
        return cls.isinstance(instance)


class Checkable(metaclass=CheckableMeta):
    @staticmethod
    @abstractmethod
    def isinstance(instance: Any) -> bool:
        ...


class Variable(Node[A], Checkable):
    collection: str

    @staticmethod
    def isinstance(instance: Any):
        return isinstance(instance, Node) and "collection" in instance.metadata


class Static(Container[A], reprlib.Representable):
    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: tp.Any) -> bool:
        return type(other) is Static and self._value == other._value

    def copy(self) -> "Static[A]":
        return Static(self._value)

    def _replace_value(
        self,
        value: B,
    ) -> "Static[B]":
        return Static(value)

    def is_equivalent(self, other: tp.Any) -> bool:
        return type(other) is Static

    def __nnx_repr__(self):
        yield reprlib.Object(type=type(self))
        yield reprlib.Attr(
            "value", repr(self._value) if isinstance(self._value, str) else self._value
        )


def _static_flatten(x: Static[tp.Any]):
    return (), x._value


def _static_unflatten(metadata: A, _) -> Static[A]:
    return Static(metadata)


jtu.register_pytree_node(Static, _static_flatten, _static_unflatten)


# --------------------
# constructors
# --------------------


def check_container(f: F, *, num_arg: int) -> F:
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        args = list(args)
        value = args[num_arg]

        if isinstance(value, Container):
            container = value
            value = container.value
        else:
            container = None

        args[num_arg] = value
        output = f(*args, **kwargs)

        if container is not None and not container.is_equivalent(output):
            raise ValueError(f"Container {container} is not equivalent to {output}")

        return output

    return wrapper  # type: ignore


@partial(check_container, num_arg=1)
def variable(
    collection: str,
    value: A,
    sharding: tp.Optional[Sharding] = None,
) -> A:
    metadata: tp.Dict[str, tp.Any] = dict(collection=collection)
    if sharding is not None:
        metadata["sharding"] = sharding
    return Node(value, **metadata)  # type: ignore


def param(
    value: A,
    sharding: tp.Optional[Sharding] = None,
) -> A:
    return variable("params", value, sharding=sharding)


@partial(check_container, num_arg=0)
def node(value: A) -> A:
    return Node(value)  # type: ignore


@partial(check_container, num_arg=0)
def static(value: A) -> A:
    return Static(value)  # type: ignore


# register nodes
nodes.register_node_type(Node)
