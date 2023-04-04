import contextlib
import functools
import typing as tp

import jax
import jax.stages
import refx
import refx.tracers
from jax._src.interpreters import pxla
from refx.partitioning import Partition

import nnx
from nnx import partitioning

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
G = tp.TypeVar("G", bound=tp.Callable[..., tp.Any])

AxisName = tp.Hashable
Leaf = tp.Any
Leaves = tp.List[Leaf]


@contextlib.contextmanager
def fork_scope_and_update_refx_trace(
    refx_trace, scope: tp.Optional["nnx.Scope"] = None
):
    if scope is None:
        scope = nnx.current_scope()
    with nnx.scope(scope.fork()), refx.tracers.refx_trace(refx_trace):
        nnx.current_scope().unsafe_trace_update()
        yield


class JitTransform(jax.stages.Wrapped):
    def __init__(
        self,
        fun: tp.Callable[..., tp.Any],
        stateful: bool,
        **jit_kwargs,
    ):
        @functools.partial(jax.jit, **jit_kwargs)
        def jitted_fn(pytree, scope: nnx.Scope, *args, **kwargs):
            top_trace = refx.tracers.current_jax_trace()
            with fork_scope_and_update_refx_trace(top_trace, scope):
                pytree = refx.reref(pytree)
                out = fun(pytree, *args, **kwargs)
                if self.stateful:
                    out = (refx.deref(pytree), out)
                return out

        self.jitted_fn = jitted_fn
        self.stateful = stateful

    def __call__(self, pytree, *args, **kwargs):
        pytree_in = pytree
        pytree = refx.deref(pytree_in)
        scope = nnx.current_scope().fork()
        out = self.jitted_fn(pytree, scope, *args, **kwargs)
        if self.stateful:
            pytree_out, out = out
            refx.update_from(pytree_in, pytree_out)
        return out

    def __repr__(self):
        return f"JitTransform({self.jitted_fn})"

    def lower(self, *args, **kwargs):
        return self.jitted_fn.lower(*args, **kwargs)


def jit(
    fun: tp.Callable[..., tp.Any],
    *,
    stateful: bool = True,
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
    ref_jit = JitTransform(
        fun,
        stateful,
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


class GradTransform:
    def __init__(
        self,
        fun: tp.Callable[..., tp.Any],
        stateful: bool,
        predicate: partitioning.Predicate,
        has_aux: bool,
        holomorphic: bool,
        allow_int: bool,
        reduce_axes: tp.Sequence[AxisName],
    ):
        @functools.partial(
            jax.grad,
            argnums=0,  # we'll handle this ourselves
            has_aux=has_aux or stateful,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        )
        def grad_fn(diff: Partition, non_diff: Partition, treedef, *args):
            diff_trace = refx.tracers.get_top_trace(diff)

            with fork_scope_and_update_refx_trace(diff_trace):
                diff, non_diff = refx.reref((diff, non_diff))
                pytree = refx.merge_partitions((diff, non_diff), treedef)
                out = fun(pytree, *args)

                if self.has_aux and self.stateful:
                    loss, aux = out
                    out = (loss, (pytree, aux))
                elif self.stateful:
                    out = (out, pytree)

                out = refx.deref(out)
                return out

        self.grad_fn = grad_fn
        self.predicate = predicate
        self.has_aux = has_aux
        self.stateful = stateful

    def __call__(self, pytree, *args):
        pytree_in = pytree
        pytree = refx.deref(pytree)
        (diff, nondiff), treedef = refx.tree_partition(pytree, self.predicate)
        # diff, nondiff = refx.deref((diff, nondiff))
        grads = self.grad_fn(diff, nondiff, treedef, *args)

        if self.has_aux and self.stateful:
            grad, (pytree_out, aux) = grads
            pytree_out, aux = refx.reref((pytree_out, aux))
            refx.update_from(pytree_in, pytree_out)
            return grad, aux
        elif self.stateful:
            grad, pytree_out = grads
            pytree_out = refx.reref(pytree_out)
            refx.update_from(pytree_in, pytree_out)
            return grad
        else:
            return grads

    def __repr__(self):
        return f"GradTransform({self.grad_fn})"


def grad(
    fun: tp.Callable[..., tp.Any],
    wrt: partitioning.CollectionFilter = "params",
    *,
    stateful: bool = True,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tp.Union[tp.Tuple[Partition, tp.Any], Partition]]:
    predicate = partitioning.to_predicate(wrt)
    ref_grad = GradTransform(
        fun,
        stateful=stateful,
        predicate=predicate,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )
    ref_grad = functools.wraps(fun)(ref_grad)
    # _update_decorator_fields(ref_grad, fun)
    return ref_grad
