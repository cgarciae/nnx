import dataclasses
from dataclasses import field
from simple_pytree import static_field
import refx
import typing as tp
import typing_extensions as tpe

A = tp.TypeVar("A")


# ----------------------------------------
# fields
# ----------------------------------------


def ref(
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
    return refx.RefField(
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
    return ref(
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


@tpe.dataclass_transform(field_specifiers=(ref, param, field, static_field))
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
