import functools
import typing as tp

import jax
import jax.stages
import refx
import refx.tracers
from jax._src import sharding_impls
from refx.partitioning import Partition
import jax.tree_util as jtu

import nnx
from nnx import partitioning, scope_lib

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
G = tp.TypeVar("G", bound=tp.Callable[..., tp.Any])

AxisName = tp.Hashable
Leaf = tp.Any
Leaves = tp.List[Leaf]


class JitTransform(jax.stages.Wrapped):
    def __init__(
        self,
        fun: tp.Callable[..., tp.Any],
        stateful: bool,
        **jit_kwargs,
    ):
        @functools.partial(jax.jit, **jit_kwargs)
        def jitted_fn(
            pytree, scope: nnx.Scope, *args, _nnx__dagdef: refx.DagDef, **kwargs
        ):
            top_trace = refx.tracers.current_jax_trace()
            with scope_lib.scope(scope.fork(), trace=top_trace):
                pytree = refx.reref(pytree, _nnx__dagdef)
                out = fun(pytree, *args, **kwargs)
                if self.stateful:
                    out = (refx.deref(pytree), out)
                return out

        self.jitted_fn = jitted_fn
        self.stateful = stateful

    def __call__(self, pytree, *args, **kwargs):
        pytree_in = pytree
        pytree, dagdef = refx.deref(pytree_in)
        scope = nnx.current_scope()
        out = self.jitted_fn(pytree, scope, *args, _nnx__dagdef=dagdef, **kwargs)
        if self.stateful:
            (pytree_out, _), out = out
            refx.update_refs(pytree_in, pytree_out)
        return out

    def __repr__(self):
        return f"JitTransform({self.jitted_fn})"

    def lower(self, *args, **kwargs):
        return self.jitted_fn.lower(*args, **kwargs)


def jit(
    fun: tp.Callable[..., tp.Any],
    *,
    stateful: bool = True,
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
        def grad_fn(
            diff: Partition,
            non_diff: Partition,
            dagdef: refx.DagDef,
            treedef: jtu.PyTreeDef,
            *args,
        ):
            diff_trace = refx.tracers.get_top_trace(diff)
            scope = scope_lib.current_scope()
            with scope_lib.scope(scope.fork(), trace=diff_trace):
                pytree = refx.merge_partitions((diff, non_diff), treedef)
                pytree = refx.reref(pytree, dagdef)
                out = fun(pytree, *args)

                pytree = refx.deref(pytree)
                if self.has_aux and self.stateful:
                    loss, aux = out
                    out = (loss, (pytree, aux))
                elif self.stateful:
                    out = (out, pytree)

                return out

        self.grad_fn = grad_fn
        self.predicate = predicate
        self.has_aux = has_aux
        self.stateful = stateful

    def __call__(self, pytree, *args):
        pytree_in = pytree
        pytree, dagdef = refx.deref(pytree)
        (diff, nondiff), treedef = refx.tree_partition(pytree, self.predicate)
        # diff, nondiff = refx.deref((diff, nondiff))
        grads = self.grad_fn(diff, nondiff, dagdef, treedef, *args)

        if self.has_aux and self.stateful:
            grad, (pytree_dagdef, aux) = grads
            pytree_out = refx.reref(*pytree_dagdef)
            refx.update_refs(pytree_in, pytree_out)
            return grad, aux
        elif self.stateful:
            grad, pytree_dagdef = grads
            pytree_out = refx.reref(*pytree_dagdef)
            refx.update_refs(pytree_in, pytree_out)
            return grad
        else:
            return grads

    def __repr__(self):
        return f"GradTransform({self.grad_fn})"


@tp.overload
def grad(
    fun: tp.Callable[..., tp.Any],
    wrt: partitioning.CollectionFilter = "params",
    *,
    stateful: bool = True,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., Partition]:
    ...


@tp.overload
def grad(
    fun: tp.Callable[..., tp.Any],
    wrt: partitioning.CollectionFilter = "params",
    *,
    stateful: bool = True,
    has_aux: tp.Literal[True],
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tp.Tuple[Partition, tp.Any]]:
    ...


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
