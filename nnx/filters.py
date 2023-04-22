import functools
import typing as tp

import jax
import jax.tree_util as jtu
import refx
import refx.tracers
from jax._src import sharding_impls
from refx.partitioning import Partition, Predicate

import nnx
from nnx import scope_lib

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
G = tp.TypeVar("G", bound=tp.Callable[..., tp.Any])

AxisName = tp.Hashable
Leaf = tp.Any
Leaves = tp.List[Leaf]


# def dagify(decorator: A, **deco_kwargs) -> A:
#     """Wraps a decorator to make it compatible with refx."""

#     @functools.wraps(decorator)
#     def decorator_wrapper(fun: F) -> F:
#         """"""

#         @functools.wraps(fun)
#         def inner_wrapper(*args, **kwargs) -> tp.Any:
#             trace = refx.tracers.get_top_trace((args, kwargs))
#             with scope_lib.scope(scope_lib.Scope.empty(), trace=trace):
#                 args, kwargs = refx.reref((args, kwargs))
#                 out = fun(*args, **kwargs)
#                 out = refx.deref(out)
#             return out

#         decorated_fun = decorator(inner_wrapper, **deco_kwargs)

#         @functools.wraps(fun)
#         def outer_wrapper(*args_in, **kwargs_in) -> tp.Any:
#             (args, kwargs), dagdef = refx.deref((args_in, kwargs_in))
#             out = decorated_fun(*args, **kwargs)
#             out = refx.reref(out)
#             return out

#         return outer_wrapper

#     return decorator_wrapper


class JitTransform(jax.stages.Wrapped):
    def __init__(
        self,
        fun: tp.Callable[..., tp.Any],
        **jit_kwargs,
    ):
        @functools.partial(jax.jit, **jit_kwargs)
        def jitted_fn(*args, _nnx__dagdef: refx.DagDef, **kwargs):
            top_trace = refx.tracers.current_jax_trace()
            with scope_lib.scope(scope_lib.Scope.empty(), trace=top_trace):
                args, kwargs = refx.reref((args, kwargs), _nnx__dagdef)
                out = fun(*args, **kwargs)
                return refx.deref(out)

        self.jitted_fn = jitted_fn

    def __call__(self, *args, **kwargs):
        (args, kwargs), dagdef = refx.deref((args, kwargs))
        out, dagdef = self.jitted_fn(*args, _nnx__dagdef=dagdef, **kwargs)
        return refx.reref(out, dagdef)

    def __repr__(self):
        return f"JitTransform({self.jitted_fn})"

    def lower(self, *args, **kwargs):
        return self.jitted_fn.lower(*args, **kwargs)


def jit_filter(
    fun: tp.Callable[..., tp.Any],
    *,
    in_shardings: tp.Any = sharding_impls.UNSPECIFIED,
    out_shardings: tp.Any = sharding_impls.UNSPECIFIED,
    static_argnums: tp.Union[int, tp.Sequence[int], None] = None,
    static_argnames: tp.Union[str, tp.Iterable[str], None] = None,
    donate_argnums: tp.Union[int, tp.Sequence[int]] = (),
    keep_unused: bool = False,
    device: tp.Optional[jax.Device] = None,
    backend: tp.Optional[str] = None,
    inline: bool = False,
    abstracted_axes: tp.Optional[tp.Any] = None,
) -> jax.stages.Wrapped:
    """JIT compile a function, dereferencing and rereferencing Refs."""

    if static_argnames is None:
        static_argnames = []
    elif isinstance(static_argnames, str):
        static_argnames = [static_argnames]
    else:
        static_argnames = list(static_argnames)

    static_argnames.append("_nnx__dagdef")

    ref_jit = JitTransform(
        fun,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
    )
    ref_jit = functools.wraps(fun)(ref_jit)
    # _update_decorator_fields(ref_jit, fun)
    return ref_jit
