import dataclasses
import typing as tp

import typing_extensions as tpe

from nnx import containers

A = tp.TypeVar("A")


def field(
    *,
    default: tp.Any = dataclasses.MISSING,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
  return dataclasses.field(  # type: ignore
      default=default,
      default_factory=default_factory,
      init=init,
      repr=repr,
      hash=hash,
      compare=compare,
      metadata=metadata,
  )


def node_field(
    *,
    default: tp.Any = dataclasses.MISSING,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
  if metadata is None:
    metadata = {}
  else:
    metadata = dict(metadata)

  if "nnx_container_fn" in metadata:
    raise ValueError("'nnx_container_fn' found in metadata")

  metadata["nnx_container_fn"] = lambda value: containers.Node(value)

  return field(
      default=default,
      default_factory=default_factory,
      init=init,
      repr=repr,
      hash=hash,
      compare=compare,
      metadata=metadata,
  )


def static_field(
    *,
    default: tp.Any = dataclasses.MISSING,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
  if metadata is None:
    metadata = {}
  else:
    metadata = dict(metadata)

  if "nnx_container_fn" in metadata:
    raise ValueError("'nnx_container_fn' found in metadata")

  metadata["nnx_container_fn"] = lambda value: containers.Static(value)

  return field(
      default=default,
      default_factory=default_factory,
      init=init,
      repr=repr,
      hash=hash,
      compare=compare,
      metadata=metadata,
  )


def var_field(
    variable_type: tp.Type[containers.Variable[tp.Any]],
    *,
    default: tp.Any = dataclasses.MISSING,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
    sharding: tp.Optional[containers.Sharding] = None,
) -> tp.Any:
  if metadata is None:
    metadata = {}
  else:
    metadata = dict(metadata)

  if "nnx_container_fn" in metadata:
    raise ValueError("'nnx_container_fn' found in metadata")

  metadata["nnx_container_fn"] = lambda value: variable_type(value, sharding=sharding)

  return field(
      default=default,
      default_factory=default_factory,
      init=init,
      repr=repr,
      hash=hash,
      compare=compare,
      metadata=metadata,
  )


def param_field(
    default: tp.Any = dataclasses.MISSING,
    *,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
  return var_field(
      containers.Param,
      default=default,
      default_factory=default_factory,
      init=init,
      repr=repr,
      hash=hash,
      compare=compare,
      metadata=metadata,
  )


@tp.overload
def dataclass(cls: tp.Type[A]) -> tp.Type[A]:
  ...


@tp.overload
def dataclass(
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> tp.Callable[[tp.Type[A]], tp.Type[A]]:
  ...


@tpe.dataclass_transform(
    field_specifiers=(field, node_field, static_field, var_field, param_field)
)
def dataclass(
    cls: tp.Optional[tp.Type[A]] = None,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> tp.Union[tp.Type[A], tp.Callable[[tp.Type[A]], tp.Type[A]]]:
  decorator = dataclasses.dataclass(
      init=init,
      repr=repr,
      eq=eq,
      order=order,
      unsafe_hash=unsafe_hash,
      frozen=frozen,
  )

  if cls is None:
    return decorator

  return decorator(cls)
