import contextlib
import typing as tp


@contextlib.contextmanager
def contexts(*contexts: tp.ContextManager[tp.Any]) -> tp.Iterator[tp.Tuple[tp.Any]]:
    """Context manager that enters multiple contexts."""
    with contextlib.ExitStack() as stack:
        yield tuple(stack.enter_context(context) for context in contexts)
