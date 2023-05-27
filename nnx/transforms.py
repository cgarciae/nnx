import functools
import typing as tp

import jax
import jax.stages
from nnx.reference import DagDef, Partition, deref, reref, update_refs
import jax.tree_util as jtu
from nnx import context

import nnx
from nnx import partitioning

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
            partition: Partition,
            *args,
            _nnx__dagdef: DagDef[tp.Any],
            **kwargs,
        ):
            pytree = reref(partition, _nnx__dagdef)
            out = fun(pytree, *args, **kwargs)
            if self.stateful:
                out = (deref(pytree), out)
            return out

        self.jitted_fn = jitted_fn
        self.stateful = stateful

    def __call__(self, pytree, *args, **kwargs):
        pytree_in = pytree

        if "ctx" in kwargs and isinstance(kwargs["ctx"], context.Context):
            kwargs["ctx"] = kwargs["ctx"].fork()

        partition, dagdef = deref(pytree_in)
        out = self.jitted_fn(partition, *args, _nnx__dagdef=dagdef, **kwargs)
        if self.stateful:
            (partition_out, _), out = out
            update_refs(pytree_in, partition_out)
        return out

    def __repr__(self):
        return f"JitTransform({self.jitted_fn})"

    def lower(self, *args, **kwargs):
        return self.jitted_fn.lower(*args, **kwargs)


UNSPECIFIED = object()


def jit(
    fun: tp.Callable[..., tp.Any],
    *,
    stateful: bool = True,
    in_shardings: tp.Any = UNSPECIFIED,
    out_shardings: tp.Any = UNSPECIFIED,
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

    jit_kwargs = dict(
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
    )

    if in_shardings is not UNSPECIFIED:
        jit_kwargs["in_shardings"] = in_shardings
    if out_shardings is not UNSPECIFIED:
        jit_kwargs["out_shardings"] = out_shardings

    ref_jit = JitTransform(
        fun,
        stateful,
        **jit_kwargs,
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
            dagdef: DagDef[tp.Any],
            *args,
        ):
            pytree = dagdef.reref((diff, non_diff))
            out = fun(pytree, *args)

            if self.stateful:
                updates = deref(pytree)[0]
                if self.has_aux:
                    loss, aux = out
                    out = (loss, (updates, aux))
                else:
                    out = (out, updates)

            return out

        self.grad_fn = grad_fn
        self.predicate = predicate
        self.has_aux = has_aux
        self.stateful = stateful

    def __call__(self, pytree, *args):
        pytree_in = pytree
        (diff, nondiff), dagdef = partitioning.tree_partition(pytree, self.predicate)
        grads = self.grad_fn(diff, nondiff, dagdef, *args)

        if self.stateful:
            updates: Partition
            if self.has_aux:
                grads, (updates, aux) = grads
                out = grads, aux
            else:
                out, updates = grads
            update_refs(pytree_in, updates)
        else:
            out = grads

        return out

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
