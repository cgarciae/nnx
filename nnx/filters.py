import functools
import typing as tp

import jax
from nnx import rng_stream
from nnx.reference import DagDef, deref, reref
from nnx.transforms import UNSPECIFIED

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
        **jit_kwargs,
    ):
        @functools.partial(jax.jit, **jit_kwargs)
        def jitted_fn(
            *args,
            _nnx__dagdef: DagDef[tp.Tuple[tp.Tuple[tp.Any, ...], tp.Dict[str, tp.Any]]],
            **kwargs,
        ):
            args, kwargs = reref((args, kwargs), _nnx__dagdef)
            out = fun(*args, **kwargs)
            return deref(out)

        self.jitted_fn = jitted_fn

    def __call__(self, *args, **kwargs):
        if "rngs" in kwargs and isinstance(kwargs["rngs"], rng_stream.Rngs):
            kwargs["rngs"] = kwargs["rngs"].fork()

        partition, dagdef = deref((args, kwargs))
        args, kwargs = dagdef.unflatten(list(partition.values()))
        out, dagdef = self.jitted_fn(*args, _nnx__dagdef=dagdef, **kwargs)
        return reref(out, dagdef)

    def __repr__(self):
        return f"JitTransform({self.jitted_fn})"

    def lower(self, *args, **kwargs):
        return self.jitted_fn.lower(*args, **kwargs)


def jit_filter(
    fun: tp.Callable[..., tp.Any],
    *,
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
        **jit_kwargs,
    )
    ref_jit = functools.wraps(fun)(ref_jit)
    # _update_decorator_fields(ref_jit, fun)
    return ref_jit
