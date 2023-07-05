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


@dataclasses.dataclass
class ContainerMetadata(tp.Generic[A]):
    value: A
    metadata: tp.Mapping[str, tp.Any]


class Container(tp.Generic[A], reprlib.Representable):
    value: A

    def __init__(self, value: tp.Union[A, ContainerMetadata[A]], **metadata: tp.Any):
        if isinstance(value, ContainerMetadata):
            metadata.update(value.metadata)
            value = tp.cast(A, value.value)

        vars(self).update(metadata, value=value)

    if tp.TYPE_CHECKING:

        def __getattr__(self, name: str) -> tp.Any:
            ...

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Container):
            return False
        return type(self) is type(other) and vars(other) == vars(self)

    @tp.overload
    def replace(self, *, value: B, **kwargs) -> "Container[B]":
        ...

    @tp.overload
    def replace(self, **kwargs) -> "Container[A]":
        ...

    def replace(self, **kwargs) -> "Container[tp.Any]":
        if "value" in kwargs:
            value = kwargs["value"]
            if isinstance(value, Container):
                if not self.is_equivalent(value):
                    raise ValueError(
                        "Cannot replace value from incompatible container, "
                        f"expected {self}, got {value}"
                    )
                kwargs["value"] = value.value

        attributes = vars(self).copy()
        # validate keys
        for key in kwargs:
            if key not in attributes:
                raise ValueError(f"Unknown metadata key {key!r}")
        attributes.update(**kwargs)
        node_type = type(self)
        return node_type(**attributes)

    def is_equivalent(self, other: tp.Any) -> bool:
        def metadata_fields(container: Container[tp.Any]) -> tp.Dict[str, tp.Any]:
            return {k: v for k, v in vars(container).items() if k != "value"}

        return type(self) is type(other) and metadata_fields(self) == metadata_fields(
            other
        )

    def copy(self: "Container[A]") -> "Container[A]":
        return type(self)(**vars(self))

    def __nnx_repr__(self):
        yield reprlib.Object(type=type(self))
        for name, value in vars(self).items():
            yield reprlib.Attr(name, repr(value))


class NodeBase(Container[A]):
    def __init_subclass__(cls):
        super().__init_subclass__()

        def _node_flatten(
            x: NodeBase[tp.Any],
            *,
            with_keys: bool,
        ):
            attributes = vars(x).copy()
            value = attributes.pop("value")
            if with_keys:
                node = (jtu.GetAttrKey("value"), value)
            else:
                node = value

            return (node,), attributes

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
    initializer: F,
    **metadata: tp.Any,
) -> F:
    @functools.wraps(initializer)
    def wrapper(*args):
        return ContainerMetadata(initializer(*args), metadata=metadata)

    return wrapper  # type: ignore


class Variable(Node[A]):
    collection: str
    sharding: tp.Optional[Sharding]

    def __init__(
        self,
        value: tp.Union[A, ContainerMetadata[A]],
        collection: str,
        sharding: tp.Optional[Sharding] = None,
        **metadata: Any,
    ):
        super().__init__(value, collection=collection, sharding=sharding, **metadata)


class Static(Container[A], reprlib.Representable):
    def __init__(self, value: A):
        super().__init__(value)

    def __hash__(self) -> int:
        return hash(self.value)


def _static_flatten(x: Static[tp.Any]):
    return (), x.value


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
