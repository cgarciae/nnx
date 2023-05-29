import dataclasses
import typing as tp
import typing_extensions as tpe

from nnx.ref_field import RefField

A = tp.TypeVar("A")


# ----------------------------------------
# fields
# ----------------------------------------

field = dataclasses.field


def ref_field(
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
    return RefField(
        collection=collection,
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
    return ref_field(
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


@tpe.dataclass_transform(field_specifiers=(ref_field, param_field, dataclasses.field))
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
