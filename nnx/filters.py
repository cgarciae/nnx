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


class FilterJIT(jax.stages.Wrapped):
    def __init__(self, fun, **jit_kwargs):
        @functools.partial(jax.jit, **jit_kwargs)
        def jitted_fn(
            *args,
            **kwargs,
        ) -> refx.Dag[tp.Tuple[A, tp.Any]]:
            args, kwargs = refx.reref((args, kwargs))
            out = fun(*args, **kwargs)
            return refx.deref(out)

        self.jitted_fn = jitted_fn

    def __call__(self, *args, **kwargs):
        args, kwargs = refx.deref((args, kwargs))
        out = self.jitted_fn(*args, **kwargs)
        out = refx.reref(out)
        return out

    def __repr__(self):
        return f"RefJIT({self.jitted_fn})"

    def lower(self, *args, **kwargs):
        return self.jitted_fn.lower(*args, **kwargs)


def filter_jit(
    fun: tp.Callable[..., tp.Any],
    in_shardings: tp.Any = pxla._UNSPECIFIED,
    out_shardings: tp.Any = pxla._UNSPECIFIED,
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
    ref_jit = FilterJIT(
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


class FilterGrad:
    def __init__(
        self,
        fun: tp.Callable[..., tp.Any],
        predicate: Predicate,
        has_aux: bool,
        holomorphic: bool,
        allow_int: bool,
        reduce_axes: tp.Sequence[AxisName],
    ):
        @functools.partial(
            jax.grad,
            argnums=0,  # we'll handle this ourselves
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        )
        def grad_fn(diff: Partition, non_diff: Partition, treedef, *args):
            # scope has been forked but it was not traced
            # so we need to manually update it
            nnx.current_scope().unsafe_trace_update()

            diff_trace = refx.tracers.get_top_trace(diff)
            with nnx.set_scope(nnx.current_scope().fork()), refx.tracers.refx_trace(
                diff_trace
            ):
                diff, non_diff = refx.reref((diff, non_diff))
                pytree = refx.merge_partitions((diff, non_diff), treedef)
                out = fun(pytree, *args)

            out = refx.deref(out)
            return out

        self.grad_fn = grad_fn
        self.predicate = predicate
        self.has_aux = has_aux

    def __call__(self, pytree, *args):
        (diff, nondiff), treedef = refx.partition_tree(pytree, self.predicate)
        diff, nondiff = refx.deref((diff, nondiff))
        grads = self.grad_fn(diff, nondiff, treedef, *args)

        if self.has_aux:
            grad, aux = grads
            aux = refx.reref(aux)
            return grad, aux
        else:
            return grads

    def __repr__(self):
        return f"RefGrad({self.grad_fn})"


def any_ref(x):
    return isinstance(x, refx.Referential)


def filter_grad(
    fun: tp.Callable[..., tp.Any],
    predicate: Predicate = any_ref,
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tp.Union[tp.Tuple[Partition, tp.Any], Partition]]:
    ref_grad = FilterGrad(
        fun,
        predicate,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )
    ref_grad = functools.wraps(fun)(ref_grad)
    # _update_decorator_fields(ref_grad, fun)
    return ref_grad
