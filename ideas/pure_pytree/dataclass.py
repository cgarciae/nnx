import dataclasses
import typing as tp
from dataclasses import field

import typing_extensions as tpe
from simple_pytree import static_field

A = tp.TypeVar("A")
K = tp.TypeVar("K", bound=tp.Hashable)


class VariableField(dataclasses.Field, tp.Generic[A]):
    def __init__(
        self,
        *,
        collection: tp.Hashable = None,
        default: tp.Any = dataclasses.MISSING,
        default_factory: tp.Any = dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
        metadata: tp.Optional[tp.Mapping[tp.Any, tp.Any]] = None,
    ):
        ...

    def __set_name__(self, cls, name):
        ...

    def __get__(self, obj, objtype=None) -> A:
        ...

    def __set__(self, obj, value: A):
        ...


# ----------------------------------------
# fields
# ----------------------------------------


def variable(
    collection: str,
    default: tp.Any = dataclasses.MISSING,
    *,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
    return VariableField(
        collection=collection,
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def param(
    default: tp.Any = dataclasses.MISSING,
    *,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
    return variable(
        "params",
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


@tpe.dataclass_transform(field_specifiers=(variable, param, field, static_field))
def dataclass(
    cls: tp.Optional[tp.Type[A]] = None,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> tp.Union[tp.Type[A], tp.Callable[[tp.Type[A]], tp.Type[A]]]:
    ...
