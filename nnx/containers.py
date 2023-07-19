import dataclasses
import functools
import typing as tp
from abc import ABCMeta
from functools import partial
from typing import Any

import jax.tree_util as jtu

from nnx import nodes, reprlib

A = tp.TypeVar("A")
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
Sharding = tp.Tuple[tp.Optional[str], ...]


@dataclasses.dataclass
class ContainerMetadata(tp.Generic[A]):
  value: A
  metadata: tp.Mapping[str, tp.Any]


class ContainerMetaclass(ABCMeta):

  def __call__(self, value: A, **metadata: tp.Any) -> A:
    if isinstance(value, Container):
      container = value
      value = container.value
    else:
      container = None

    obj = super().__call__(value, **metadata)

    if container is not None and not container.is_equivalent(obj):
      raise ValueError(
          f"input value of type '{type(container).__name__}' is not compatible "
          f"with return type '{type(obj).__name__}'"
      )

    return obj


class Container(tp.Generic[A], reprlib.Representable, metaclass=ContainerMetaclass):
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

    return type(self) is type(other) and metadata_fields(self) == metadata_fields(other)

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


class Variable(Node[A]):
  sharding: tp.Optional[Sharding]

  def __init__(
      self,
      value: tp.Union[A, ContainerMetadata[A]],
      sharding: tp.Optional[Sharding] = None,
      **metadata: Any,
  ):
    super().__init__(value, sharding=sharding, **metadata)


class Param(Variable[A]):
  pass


class BatchStat(Variable[A]):
  pass


class Cache(Variable[A]):
  pass


class Intermediate(Variable[A]):
  pass


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


def with_metadata(
    initializer: F,
    **metadata: tp.Any,
) -> F:
  @functools.wraps(initializer)
  def wrapper(*args):
    return ContainerMetadata(initializer(*args), metadata=metadata)

  return wrapper  # type: ignore


# register nodes
nodes.register_node_type(Node)
