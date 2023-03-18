import dataclasses
import refx
import typing as tp

A = tp.TypeVar("A")


# ----------------------------------------
# Refs
# ----------------------------------------


class BatchStat(refx.Ref[A]):
    pass


class Param(refx.Ref[A]):
    pass


# ----------------------------------------
# fields
# ----------------------------------------


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
    return refx.ref_field(
        default=default,
        ref_type=Param,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def batch_stat(
    default: tp.Any = dataclasses.MISSING,
    *,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
    return refx.ref_field(
        default=default,
        ref_type=BatchStat,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )
