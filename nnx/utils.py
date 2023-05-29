import typing as tp

A = tp.TypeVar("A")


def first_from(*args: tp.Optional[A]) -> A:
    """Return the first non-None argument."""
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError("No non-None arguments found.")
