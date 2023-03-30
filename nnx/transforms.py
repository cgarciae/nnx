import functools
import typing as tp
import jax
import jax.stages
import refx
from jax._src.interpreters import pxla

from nnx.ref_fields import Param

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
AxisName = tp.Hashable
TypeOrSeqType = tp.Union[
    tp.Type[refx.Ref[tp.Any]], tp.Sequence[tp.Type[refx.Ref[tp.Any]]]
]
Leaf = tp.Any
Leaves = tp.List[Leaf]


class RefJIT(jax.stages.Wrapped):
    def __init__(self, fun, **jit_kwargs):
        @functools.partial(jax.jit, **jit_kwargs)
        def jitted_fn(
            dag: refx.Dag[A],
            *args,
            **kwargs,
        ) -> refx.Dag[tp.Tuple[A, tp.Any]]:
            pytree = dag.value
            out = fun(pytree, *args, **kwargs)
            return refx.Dag((pytree, out))

        self.jitted_fn = jitted_fn

    def __call__(self, pytree, *args, **kwargs):
        dag = self.jitted_fn(refx.Dag(pytree), *args, **kwargs)
        pytree_out, out = dag.value
        refx.update_from(pytree, pytree_out)
        return out

    def __repr__(self):
        return f"RefJIT({self.jitted_fn})"

    def lower(self, *args, **kwargs):
        return self.jitted_fn.lower(*args, **kwargs)


def jit(
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
    ref_jit = RefJIT(
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


class RefGrad:
    def __init__(
        self,
        fun: tp.Callable[..., tp.Any],
        type_predicate: TypeOrSeqType,
        **grad_kwargs,
    ):
        @functools.partial(jax.grad, **grad_kwargs)
        def grad_fn(diff, non_diff, treedef, *args, **kwargs):
            diff, non_diff = refx.reref((diff, non_diff))
            pytree = refx.merge_partitions((diff, non_diff), treedef)
            out = fun(pytree, *args, **kwargs)
            out = refx.deref(out)
            return out

        self.grad_fn = grad_fn
        self.type_predicate = type_predicate
        self.has_aux: bool = grad_kwargs["has_aux"]

    def __call__(self, pytree, *args, **kwargs):
        (diff, non_diff), treedef = refx.partition_tree(pytree, self.type_predicate)
        diff, non_diff = refx.deref((diff, non_diff))
        grads = self.grad_fn(diff, non_diff, treedef, *args, **kwargs)

        if self.has_aux:
            grad, aux = grads
            aux = refx.reref(aux)
            return grad, aux
        else:
            return grads

    def __repr__(self):
        return f"RefGrad({self.grad_fn})"


@overload
def grad(
    fun: tp.Callable[..., tp.Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    has_aux: Literal[False] = False,
    argnums: tp.Union[int, tp.Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tp.Any]:
    ...


@overload
def grad(
    fun: tp.Callable[..., tp.Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    has_aux: Literal[True],
    argnums: tp.Union[int, tp.Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., Tuple[tp.Any, tp.Any]]:
    ...


def grad(
    fun: tp.Callable[..., tp.Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    argnums: tp.Union[int, tp.Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tp.Union[Tuple[tp.Any, tp.Any], tp.Any]]:
    ref_grad = RefGrad(
        fun,
        type_predicate,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )
    ref_grad = functools.wraps(fun)(ref_grad)
    # _update_decorator_fields(ref_grad, fun)
    return ref_grad
