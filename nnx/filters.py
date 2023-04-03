import functools
import typing as tp

import jax
import jax.tree_util as jtu
import refx
import refx.tracers
import nnx
from jax._src.interpreters import pxla
from refx.partitioning import Partition, Predicate

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
G = tp.TypeVar("G", bound=tp.Callable[..., tp.Any])

AxisName = tp.Hashable
Leaf = tp.Any
Leaves = tp.List[Leaf]


def dagify(decorator: A, propagate_state: bool = False, **deco_kwargs) -> A:
    """Wraps a decorator to make it compatible with refx."""

    @functools.wraps(decorator)
    def decorator_wrapper(fun: F) -> F:
        """"""

        @functools.wraps(fun)
        def inner_wrapper(*args, **kwargs) -> tp.Any:
            args, kwargs = refx.reref((args, kwargs))
            out = fun(*args, **kwargs)
            if propagate_state:
                out = refx.deref(((args, kwargs), out))
            else:
                out = refx.deref(out)
            return out

        decorated_fun = decorator(inner_wrapper, **deco_kwargs)

        @functools.wraps(fun)
        def outer_wrapper(*args_in, **kwargs_in) -> tp.Any:
            args, kwargs = refx.deref((args_in, kwargs_in))
            out = decorated_fun(*args, **kwargs)
            out = refx.reref(out)
            if propagate_state:
                (args_out, kwargs_out), out = out
                refx.update_from((args_in, kwargs_in), (args_out, kwargs_out))
            return out

        return outer_wrapper

    return decorator_wrapper
