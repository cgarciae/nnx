import functools
import typing as tp

import jax
import jax.stages
from nnx.module import AnySplit, Module, ModuleDef, Split
from nnx.state import State
from nnx import context

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
        def jitted_fn(module: Module, *args, **kwargs):
            out = fun(module, *args, **kwargs)
            if self.stateful:
                out = (module, out)
            return out

        self.jitted_fn = jitted_fn
        self.stateful = stateful

    def __call__(self, module: Module, *args, **kwargs):
        if "ctx" in kwargs and isinstance(kwargs["ctx"], context.Context):
            kwargs["ctx"] = kwargs["ctx"].fork()

        out = self.jitted_fn(module, *args, **kwargs)
        if self.stateful:
            updates, out = out
            module.update(updates)
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
            diff: State,
            non_diff: State,
            moddef: ModuleDef[Module],
            is_module,
            *args: tp.Any,
        ):
            if is_module:
                mod_or_split = moddef.merge((diff, non_diff))
            else:
                mod_or_split = Split(((diff, non_diff), moddef))
            out = fun(mod_or_split, *args)

            if self.stateful and is_module:
                updates = mod_or_split.split(...).states
                if self.has_aux:
                    loss, aux = out
                    out = (loss, (updates, aux))
                else:
                    out = (out, updates)
            elif self.stateful and not is_module and not self.has_aux:
                out = (out, None)

            return out

        self.grad_fn = grad_fn
        self.predicate = predicate
        self.has_aux = has_aux
        self.stateful = stateful

    def __call__(self, mod_or_split: tp.Union[Module, AnySplit[Module]], *args: tp.Any):
        if not isinstance(mod_or_split, (Module, Split)):
            raise TypeError(
                f"Expected a Module or Split, got {type(mod_or_split).__name__}"
            )

        is_module = isinstance(mod_or_split, Module)

        (diff, nondiff), moddef = mod_or_split.split(self.predicate, ...)
        grads = self.grad_fn(diff, nondiff, moddef, is_module, *args)

        if self.stateful and is_module:
            updates: State
            if self.has_aux:
                grads, (updates, aux) = grads
                out = grads, aux
            else:
                out, updates = grads
            mod_or_split.update(updates)
        elif self.stateful and not is_module and not self.has_aux:
            out, _ = grads
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
) -> tp.Callable[..., State]:
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
) -> tp.Callable[..., tp.Tuple[State, tp.Any]]:
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
) -> tp.Callable[..., tp.Union[tp.Tuple[State, tp.Any], State]]:
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
