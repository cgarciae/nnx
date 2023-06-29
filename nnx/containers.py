import dataclasses
import functools
import typing as tp
from abc import ABC, abstractmethod
from functools import partial
from types import MappingProxyType

import jax.tree_util as jtu

from nnx import nodes, reprlib
from nnx.nn import initializers

A = tp.TypeVar("A")
B = tp.TypeVar("B")
C = tp.TypeVar("C", bound="Container[tp.Any]")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
Sharding = tp.Tuple[str, ...]


@dataclasses.dataclass
class ContainerMetadata(tp.Generic[A]):
    value: A
    metadata: tp.Mapping[str, tp.Hashable]


class Container(tp.Generic[A], reprlib.Representable):
    __slots__ = ("_value", "_metadata")

    def __init__(
        self,
        value: tp.Union[A, ContainerMetadata[A]],
        metadata: tp.Mapping[str, tp.Hashable],
    ):
        if isinstance(value, ContainerMetadata):
            metadata = {**value.metadata, **metadata}
            value = tp.cast(A, value.value)

        self._value = value
        self._metadata = MappingProxyType(metadata)

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

    def _replace_value(self: "Container[A]", value: B) -> "Container[B]":
        return Container(value, self._metadata)

    def is_equivalent(self, other: tp.Any) -> bool:
        return type(other) is Container and self._metadata == other._metadata

    def copy(self: "Container[A]") -> "Container[A]":
        return Container(self._value, self._metadata)

    def __nnx_repr__(self):
        yield reprlib.Object(type=type(self))
        yield reprlib.Attr(
            "value", repr(self._value) if isinstance(self._value, str) else self._value
        )
        yield reprlib.Attr("metadata", self._metadata)


def _container_flatten(
    x: Container[tp.Any],
    *,
    with_keys: bool,
):
    if with_keys:
        node = (jtu.GetAttrKey("value"), x._value)
    else:
        node = x._value

    return (node,), x._metadata


def _container_unflatten(
    metadata: tp.Mapping[str, tp.Hashable], children: tp.Tuple[A]
) -> Container[A]:
    return Container(children[0], metadata)


jtu.register_pytree_with_keys(
    Container,
    partial(_container_flatten, with_keys=True),
    _container_unflatten,
    flatten_func=partial(_container_flatten, with_keys=False),
)


def with_partitioning(
    initializer: initializers.Initializer,
    sharding: Sharding,
) -> initializers.Initializer:
    @functools.wraps(initializer)
    def wrapper(*args):
        return ContainerMetadata(initializer(*args), metadata={"sharding": sharding})

    return wrapper  # type: ignore


class Variable(Container[A], reprlib.Representable):
    __slots__ = ("_collection", "_sharding")

    def __init__(
        self,
        value: A,
        collection: str,
        sharding: tp.Optional[Sharding],
    ):
        if isinstance(value, ContainerMetadata):
            sharding = value.sharding if sharding is None else sharding
            value = tp.cast(A, value.value)

        self._value = value
        self._collection = collection
        self._sharding = sharding

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is Variable
            and self._value == other._value
            and self._collection == other._collection
            and self._sharding == other._sharding
        )

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def sharding(self) -> tp.Optional[Sharding]:
        return self._sharding

    def __nnx_repr__(self):
        yield reprlib.Object(type=type(self))
        yield reprlib.Attr("collection", repr(self._collection))
        yield reprlib.Attr(
            "value", repr(self._value) if isinstance(self._value, str) else self._value
        )
        if self._sharding is not None:
            yield reprlib.Attr("sharding", self._sharding)

    def copy(self) -> "Variable[A]":
        return Variable(self._value, self._collection, self._sharding)

    def _replace_value(
        self,
        value: B,
    ) -> "Variable[B]":
        return Variable(
            value,
            self._collection,
            self._sharding,
        )

    def is_equivalent(self, other: tp.Any) -> bool:
        return (
            type(other) is Variable
            and self._collection == other._collection
            and self._sharding == other._sharding
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
def var(
    collection: str,
    value: A,
    sharding: tp.Optional[Sharding] = None,
) -> A:
    return Variable(  # type: ignore
        value,
        collection=collection,
        sharding=sharding,
    )


def param(
    value: A,
    sharding: tp.Optional[Sharding] = None,
) -> A:
    return var("params", value, sharding=sharding)


@partial(check_container, num_arg=0)
def node(value: A) -> A:
    return Node(value)  # type: ignore


@partial(check_container, num_arg=0)
def static(value: A) -> A:
    return Static(value)  # type: ignore


# register nodes
nodes.register_node_type(Node)
nodes.register_node_type(Variable)
