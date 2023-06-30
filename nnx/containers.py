import dataclasses
import functools
import typing as tp
from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import jax.tree_util as jtu

from nnx import nodes, reprlib
from nnx.nn import initializers

A = tp.TypeVar("A")
B = tp.TypeVar("B")
C = tp.TypeVar("C", bound="Container[tp.Any]")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
TNodeBase = tp.TypeVar("TNodeBase", bound="NodeBase[tp.Any]")
Sharding = tp.Tuple[tp.Optional[str], ...]


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


class NodeBase(Container[A], reprlib.Representable):
    def __init__(
        self,
        value: tp.Union[A, NodeMetadata[A]],
        **metadata: tp.Any,
    ):
        if isinstance(value, NodeMetadata):
            metadata.update(value.metadata)
            value = tp.cast(A, value.value)

        self._value = value
        vars(self).update(metadata)

    if tp.TYPE_CHECKING:

        def __getattr__(self, name: str) -> tp.Any:
            ...

    def _replace_value(self: TNodeBase, value: B) -> TNodeBase:
        node_type = type(self)
        return node_type(value, **vars(self))

    def replace_metadata(self: TNodeBase, **kwargs) -> TNodeBase:
        metadata = vars(self).copy()
        # validate keys
        for key in kwargs:
            if key not in metadata:
                raise ValueError(f"Unknown metadata key {key!r}")
        metadata.update(**kwargs)
        node_type = type(self)
        return node_type(self.value, **metadata)

    def is_equivalent(self, other: tp.Any) -> bool:
        return type(other) is type(self) and vars(other) == vars(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeBase):
            return False
        return (
            type(self) is type(other)
            and self.value == other.value
            and vars(other) == vars(self)
        )

    def copy(self: TNodeBase) -> TNodeBase:
        node_type = type(self)
        return node_type(self._value, **vars(self))

    def __nnx_repr__(self):
        yield reprlib.Object(type=type(self))
        yield reprlib.Attr(
            "value", repr(self._value) if isinstance(self._value, str) else self._value
        )
        for name, value in vars(self).items():
            yield reprlib.Attr(name, repr(value))

    def __init_subclass__(cls):
        super().__init_subclass__()

        def _node_flatten(
            x: NodeBase[tp.Any],
            *,
            with_keys: bool,
        ):
            if with_keys:
                node = (jtu.GetAttrKey("value"), x._value)
            else:
                node = x._value

            return (node,), vars(x).copy()

        def _node_unflatten(
            metadata: tp.Mapping[str, tp.Any], children: tp.Tuple[A]
        ) -> NodeBase[A]:
            return cls(children[0], **metadata)

        jtu.register_pytree_with_keys(
            cls,
            partial(_node_flatten, with_keys=True),
            _node_unflatten,
            flatten_func=partial(_node_flatten, with_keys=False),
        )


class Node(NodeBase[A]):
    pass


def with_metadata(
    initializer: initializers.Initializer,
    **metadata: tp.Any,
) -> initializers.Initializer:
    @functools.wraps(initializer)
    def wrapper(*args):
        return NodeMetadata(initializer(*args), metadata=metadata)

    return wrapper  # type: ignore


class Variable(Node[A]):
    collection: str
    sharding: tp.Optional[Sharding]

    def __init__(
        self,
        value: tp.Union[A, NodeMetadata[A]],
        collection: str,
        sharding: tp.Optional[Sharding] = None,
        **metadata: Any,
    ):
        super().__init__(value, collection=collection, sharding=sharding, **metadata)


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


@partial(check_container, num_arg=0)
def node(value: A, **metadata: tp.Any) -> A:
    return Node(value, **metadata)  # type: ignore


@partial(check_container, num_arg=1)
def variable(
    collection: str,
    value: A,
    sharding: tp.Optional[Sharding] = None,
    **metadata: tp.Any,
) -> A:
    return Variable(
        value,
        collection=collection,
        sharding=sharding,
        **metadata,
    )  # type: ignore


def param(value: A, sharding: tp.Optional[Sharding] = None, **metadata: tp.Any) -> A:
    return variable("params", value, sharding=sharding, **metadata)


@partial(check_container, num_arg=0)
def static(value: A) -> A:
    return Static(value)  # type: ignore


# register nodes
nodes.register_node_type(Node)
