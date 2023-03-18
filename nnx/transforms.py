import functools
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
import jax
import jax.stages
import refx
from jax._src.interpreters import pxla

from nnx.refs import Param

A = TypeVar("A")
F = TypeVar("F", bound=Callable[..., Any])
AxisName = Hashable
TypeOrSeqType = Union[Type[refx.Ref[Any]], Sequence[Type[refx.Ref[Any]]]]


class RefJIT(jax.stages.Wrapped):
    def __init__(self, fun, **jit_kwargs):
        @functools.partial(jax.jit, **jit_kwargs)
        def jitted_fn(*args, **kwargs):
            args, kwargs = refx.reref((args, kwargs))
            out = fun(*args, **kwargs)
            out = refx.deref(out)
            return out

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


def _update_decorator_fields(
    decorator: Callable[..., Any], wrapped: Callable[..., Any]
):
    """Update the fields of a decorator to match the wrapped function."""
    if hasattr(wrapped, "__signature__"):
        decorator.__signature__ = wrapped.__signature__
        # decorator.__call__.__signature__ = wrapped.__signature__
    if hasattr(wrapped, "__name__"):
        decorator.__name__ = wrapped.__name__
        # decorator.__call__.__name__ = wrapped.__name__


def jit(
    fun: Callable[..., Any],
    in_shardings: Any = pxla._UNSPECIFIED,
    out_shardings: Any = pxla._UNSPECIFIED,
    static_argnums: Union[int, Sequence[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    donate_argnums: Union[int, Sequence[int]] = (),
    keep_unused: bool = False,
    device: Optional[jax.Device] = None,
    backend: Optional[str] = None,
    inline: bool = False,
    abstracted_axes: Optional[Any] = None,
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
        fun: Callable[..., Any],
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
    fun: Callable[..., Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    has_aux: Literal[False] = False,
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., Any]:
    ...


@overload
def grad(
    fun: Callable[..., Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    has_aux: Literal[True],
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., Tuple[Any, Any]]:
    ...


def grad(
    fun: Callable[..., Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., Union[Tuple[Any, Any], Any]]:
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
