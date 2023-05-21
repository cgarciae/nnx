import contextlib
import typing as tp
import inspect

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


def has_kwarg(fn: tp.Callable[..., tp.Any], kwarg: str) -> bool:
    """Return True if the function has the given keyword argument."""
    parameters = inspect.signature(fn).parameters
    return (
        kwarg in parameters and parameters[kwarg].kind == inspect.Parameter.KEYWORD_ONLY
    )
