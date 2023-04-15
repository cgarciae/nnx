import contextlib
import typing as tp

A = tp.TypeVar("A")


@contextlib.contextmanager
def contexts(*contexts: tp.ContextManager[tp.Any]) -> tp.Iterator[tp.Tuple[tp.Any]]:
    """Context manager that enters multiple contexts."""
    with contextlib.ExitStack() as stack:
        yield tuple(stack.enter_context(context) for context in contexts)


def first_from(*args: tp.Optional[A]) -> A:
    """Return the first non-None argument."""
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError("No non-None arguments found.")
