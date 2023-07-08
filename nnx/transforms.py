import functools
import typing as tp
from types import MappingProxyType

import jax
import jax.numpy as jnp
import jax.stages
import jax.tree_util as jtu

from nnx import containers, contextlib, partitioning, spmd, tracers
from nnx.module import CallableProxy, DelayedAccessor, Module, ModuleDef, PureModule
from nnx.state import State

A = tp.TypeVar("A")
C = tp.TypeVar("C")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
G = tp.TypeVar("G", bound=tp.Callable[..., tp.Any])
M = tp.TypeVar("M", bound=Module)

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
        def jitted_fn(pure_module: PureModule[Module], *args, **kwargs):
            if "ctx" in kwargs and isinstance(kwargs["ctx"], contextlib.PureContext):
                kwargs["ctx"] = kwargs["ctx"].merge()

            nnx_trace = tracers.get_top_trace((args, kwargs))
            with tracers.nnx_trace(nnx_trace):
                module = pure_module.merge()
                out = fun(module, *args, **kwargs)

            if self.stateful:
                updates = module.get_state()
                out = (updates, out)

            return out

        self.jitted_fn = jitted_fn
        self.stateful = stateful

    def __call__(self, module: tp.Any, *args, **kwargs):
        if not isinstance(module, Module):
            raise TypeError(f"Expected Module, got {type(module).__name__}")
        if "ctx" in kwargs and isinstance(kwargs["ctx"], contextlib.Context):
            kwargs["ctx"] = kwargs["ctx"].partition()

        pure_module = module.partition()
        out = self.jitted_fn(pure_module, *args, **kwargs)
        if self.stateful:
            updates: State
            updates, out = out
            module.update_state(updates)
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
            moduledef: ModuleDef[Module],
            *args: tp.Any,
        ):
            with tracers.nnx_trace(tracers.get_top_trace(diff)):
                module = moduledef.merge(diff, non_diff)
                out = fun(module, *args)

            if self.stateful:
                updates = module.get_state()
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

    def __call__(self, module: Module, *args: tp.Any):
        if not isinstance(module, Module):
            raise TypeError(f"Expected a Module, got {type(module).__name__}")

        (diff, nondiff), moduledef = module.partition(self.predicate, ...)

        grads = self.grad_fn(diff, nondiff, moduledef, *args)

        if self.stateful:
            updates: State
            if self.has_aux:
                grads, (updates, aux) = grads
                out = grads, aux
            else:
                out, updates = grads
            module.update_state(updates)
        else:
            out = grads

        return out

    def __repr__(self):
        return f"GradTransform({self.grad_fn})"


@tp.overload
def grad(
    fun: tp.Callable[..., tp.Any],
    wrt: partitioning.Filter = "params",
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
    wrt: partitioning.Filter = "params",
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
    wrt: partitioning.Filter = containers.Param,
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


# NOTE: I don't understand why variable_broadcasts exists. Passing them as
# captures to `scan_fn` makes it impossible to propagate state updates from
# them. Maybe, there should only be `variable_carry` and `variable_axes`.
class Scan(Module, tp.Generic[M]):
    def __init__(
        self,
        *,
        module_constructor: tp.Callable[..., M],
        module_init_args: tp.Tuple[tp.Any, ...],
        module_init_kwargs: tp.Dict[str, tp.Any],
        variable_axes: tp.Mapping[partitioning.Filter, int],
        variable_broadcast: partitioning.Filter,
        variable_carry: partitioning.Filter,
        split_rngs: contextlib.RngFilter,
        in_axes: tp.Any,
        out_axes: tp.Any,
        length: tp.Optional[int],
        reverse: bool,
        unroll: int,
        data_transform: tp.Optional[tp.Callable[..., tp.Any]],
        metadata_params: tp.Mapping[tp.Any, tp.Any],
    ):
        self.module_constructor = module_constructor
        self.variable_axes = variable_axes
        self.variable_broadcast = variable_broadcast
        self.variable_carry = variable_carry
        self.split_rngs = split_rngs
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.length = length
        self.reverse = reverse
        self.unroll = unroll
        self.data_transform = data_transform
        self.metadata_params = metadata_params
        self.scan_module = self.create_module(*module_init_args, **module_init_kwargs)

    def create_module(
        self: "Scan[M]", *args, ctx: tp.Optional[contextlib.Context] = None, **kwargs
    ) -> M:
        if self.variable_axes and self.length is None:
            raise ValueError("Cannot use variable_axes without specifying a length")

        key_values = []

        if ctx is not None:
            if not isinstance(ctx, contextlib.Context):
                raise TypeError(f"Expected a Context, got {type(ctx).__name__}")

            keys, ctxdef = ctx.partition()
            split_predicate = contextlib.to_rng_predicate(self.split_rngs)

            key_axes = []
            key_names = tuple(keys.keys())

            for name, key in keys.items():
                if split_predicate(name):
                    if self.length is None:
                        raise ValueError(
                            "Cannot split RNGs without specifying a length"
                        )
                    key = jax.random.split(key, self.length)
                    key_axes.append(0)
                else:
                    key_axes.append(None)
                key_values.append(key)
        else:
            key_names = None
            ctxdef = None
            key_axes = None

        init_out_axes = (*self.variable_axes.values(), None, None)
        moduledef: tp.Optional[ModuleDef[M]] = None

        def _init_state(*key_values):
            nonlocal moduledef

            if ctxdef is not None:
                assert key_names is not None
                keys = dict(zip(key_names, key_values))
                ctx = ctxdef.merge(keys)
                kwargs["ctx"] = ctx

            module = self.module_constructor(*args, **kwargs)

            # lift module
            filters = (
                *self.variable_axes.keys(),
                self.variable_broadcast,
                self.variable_carry,
            )

            states, moduledef = module.partition(*filters)

            return states

        if ctxdef is not None or self.variable_axes:
            _init_state = jax.vmap(
                _init_state,
                in_axes=key_axes,
                out_axes=init_out_axes,
                axis_size=self.length,
            )

        *axes_states, broadcast_state, carry_state = _init_state(*key_values)
        moduledef = tp.cast(ModuleDef[M], moduledef)

        # add additional axis name to Variable.sharding
        if spmd.PARTITION_NAME in self.metadata_params:
            axes_states = [
                spmd.add_axis(state, index, self.metadata_params)
                for state, index in zip(axes_states, self.variable_axes.values())
            ]

        module = moduledef.merge(*axes_states, broadcast_state, carry_state)

        return module

    def __call__(
        self,
        carry_arg: C,
        axes_arg,
        *broadcast_args,
        ctx: tp.Optional[contextlib.Context] = None,
        **broadcast_kwargs,
    ) -> tp.Tuple[C, tp.Any]:
        return self.call(  # type: ignore
            carry_arg, axes_arg, *broadcast_args, ctx=ctx, **broadcast_kwargs
        )

    @property
    def call(self) -> M:
        accessesor = DelayedAccessor()

        def _context(
            accessesor,
            carry_arg: C,
            axes_arg,
            *broadcast_args,
            ctx: tp.Optional[contextlib.Context] = None,
            **broadcast_kwargs,
        ) -> tp.Tuple[C, tp.Any]:
            # split module state
            filters = (
                *self.variable_axes.keys(),
                self.variable_broadcast,
                self.variable_carry,
            )
            (
                *axes_states,
                broadcast_state,
                carry_state,
            ), moduledef = self.scan_module.partition(*filters)

            # transpose axes state
            axes_states = tuple(
                jax.tree_map(lambda x: jnp.moveaxis(x, axis, 0), axes_state)
                for axes_state, axis in zip(axes_states, self.variable_axes.values())
            )
            # transpose axes arg
            axes_arg = tree_map_upto_left(
                lambda axis, node: jax.tree_map(
                    lambda x: jnp.moveaxis(x, axis, 0), node
                ),
                self.in_axes,
                axes_arg,
            )

            # infer length
            lengths: tp.Set[int] = set(
                x.shape[0] for x in jax.tree_util.tree_leaves((axes_states, axes_arg))
            )

            if len(lengths) > 1:
                raise ValueError(
                    f"Inconsistent lengths between variable_axes states and "
                    f"axes_arg: {lengths}"
                )
            elif len(lengths) == 0:
                if self.length is None:
                    raise ValueError(
                        "Cannot infer length from variable_axes states or axes_arg, "
                        "please specify `length`"
                    )
                length = self.length
            else:
                length = lengths.pop()
                if self.length is not None and self.length != length:
                    raise ValueError(
                        f"Specified length {self.length} is the same as the inferred "
                        f"length {length}"
                    )

            # split rng state
            axes_keys: tp.Optional[tp.Dict[str, jax.Array]]
            broadcast_keys: tp.Optional[tp.Dict[str, jax.Array]]

            if ctx is not None:
                if not isinstance(ctx, contextlib.Context):
                    raise TypeError(f"Expected a Context, got {type(ctx).__name__}")

                axes_keys = {}
                broadcast_keys = {}

                keys, ctxdef = ctx.partition()
                split_predicate = contextlib.to_rng_predicate(self.split_rngs)

                for name, key in keys.items():
                    if split_predicate(name):
                        axes_keys[name] = jax.random.split(key, length)
                    else:
                        broadcast_keys[name] = key
            else:
                ctxdef = None
                axes_keys = None
                broadcast_keys = None

            carry = (carry_state, carry_arg)
            axes = (axes_keys, axes_states, axes_arg)

            def scan_fn(
                carry: tp.Tuple[State, tp.Any],
                axes: tp.Tuple[
                    tp.Optional[tp.Dict[str, jax.Array]], tp.Tuple[State, ...], tp.Any
                ],
            ):
                carry_state, carry_arg = carry
                axes_keys, axes_states, axes_arg = axes

                # merge rng state
                if ctxdef is not None:
                    assert axes_keys is not None and broadcast_keys is not None
                    ctx = ctxdef.merge({**axes_keys, **broadcast_keys})
                    broadcast_kwargs["ctx"] = ctx

                # remove metadata axis name from Variable.sharding
                if spmd.PARTITION_NAME in self.metadata_params:
                    axes_states = [
                        spmd.remove_axis(state, index, self.metadata_params)
                        for state, index in zip(
                            axes_states, self.variable_axes.values()
                        )
                    ]

                # merge module state
                module = moduledef.merge(*axes_states, broadcast_state, carry_state)

                fn = accessesor(module)
                (carry_out, axes_out) = fn(
                    carry_arg, axes_arg, *broadcast_args, **broadcast_kwargs
                )

                # split module state
                (*axes_states, _broadcast_state, carry_state), _ = module.partition(
                    *filters
                )

                # add metadata axis name to Variable.sharding
                if spmd.PARTITION_NAME in self.metadata_params:
                    axes_states = [
                        spmd.add_axis(state, index, self.metadata_params)
                        for state, index in zip(
                            axes_states, self.variable_axes.values()
                        )
                    ]

                carry = (carry_state, carry_out)
                out = (axes_states, axes_out)

                return carry, out

            carry, scan_out = jax.lax.scan(
                scan_fn,
                carry,
                axes,
                length=length,
                reverse=self.reverse,
                unroll=self.unroll,
            )
            carry_state, carry_out = carry
            axes_states, out = scan_out

            # transpose axes state
            axes_states = tuple(
                jax.tree_map(lambda x: jnp.moveaxis(x, 0, axis), axes_state)
                for axes_state, axis in zip(axes_states, self.variable_axes.values())
            )
            # transpose axes arg
            out = tree_map_upto_left(
                lambda axis, node: jax.tree_map(
                    lambda x: jnp.moveaxis(x, 0, axis), node
                ),
                self.out_axes,
                out,
            )

            self.scan_module.update_state((*axes_states, carry_state))

            return carry_out, out

        return CallableProxy(_context, accessesor)  # type: ignore


def scan(
    module_constructor: tp.Callable[..., M],
    variable_axes: tp.Mapping[partitioning.Filter, int] = MappingProxyType({}),
    variable_broadcast: partitioning.Filter = None,
    variable_carry: partitioning.Filter = ...,
    split_rngs: contextlib.RngFilter = None,
    in_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    length: tp.Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
    data_transform: tp.Optional[tp.Callable[..., tp.Any]] = None,
    metadata_params: tp.Mapping[tp.Any, tp.Any] = {},
) -> tp.Callable[..., Scan[M]]:
    def _create_scan(*args, **kwargs) -> Scan[M]:
        return Scan(
            module_constructor=module_constructor,
            module_init_args=args,
            module_init_kwargs=kwargs,
            variable_axes=variable_axes,
            variable_broadcast=variable_broadcast,
            variable_carry=variable_carry,
            split_rngs=split_rngs,
            in_axes=in_axes,
            out_axes=out_axes,
            length=length,
            reverse=reverse,
            unroll=unroll,
            data_transform=data_transform,
            metadata_params=metadata_params,
        )

    return _create_scan


class Remat(Module, tp.Generic[M]):
    def __init__(
        self,
        *,
        module_constructor: tp.Callable[..., M],
        module_init_args: tp.Tuple[tp.Any, ...],
        module_init_kwargs: tp.Dict[str, tp.Any],
        # variables: lift.CollectionFilter,
        # rngs: lift.PRNGSequenceFilter,
        prevent_cse: bool,
        static_argnums: tp.Union[int, tuple[int, ...]],
        policy: tp.Optional[tp.Callable[..., bool]],
    ):
        if isinstance(static_argnums, int):
            static_argnums = (static_argnums,)

        # add 2 as an offset to account for state and keys
        static_argnums = tuple(x + 2 if x >= 0 else x for x in static_argnums)

        self.module_constructor = module_constructor
        self.prevent_cse = prevent_cse
        self.static_argnums = static_argnums
        self.policy = policy
        self.remat_module = self.module_constructor(
            *module_init_args, **module_init_kwargs
        )

    def __call__(
        self,
        *args,
        ctx: tp.Optional[contextlib.Context] = None,
    ):
        return self.call(*args, ctx=ctx)  # type: ignore

    @property
    def call(self) -> M:
        accessesor = DelayedAccessor()

        def _call(
            accessesor, *args, ctx: tp.Optional[contextlib.Context] = None
        ) -> tp.Tuple[tp.Any]:
            state, moduledef = self.remat_module.partition()

            if ctx is not None:
                keys, ctxdef = ctx.partition()
            else:
                keys = None
                ctxdef = None

            def _remat_fn(
                state: State,
                keys: tp.Optional[tp.Dict[str, jax.Array]],
                *args,
            ) -> tp.Tuple[State, tp.Any]:
                kwargs = {}
                if keys is not None:
                    assert ctxdef is not None
                    kwargs["ctx"] = ctxdef.merge(keys)

                module = moduledef.merge(state)
                fn = accessesor(module)
                out = fn(*args, **kwargs)

                state, _ = module.partition()

                return state, out

            state, out = jax.checkpoint(
                _remat_fn,
                prevent_cse=self.prevent_cse,
                static_argnums=self.static_argnums,
                policy=self.policy,
            )(state, keys, *args)

            self.remat_module.update_state(state)

            return out

        return CallableProxy(_call, accessesor)  # type: ignore


def remat(
    module_constructor: tp.Callable[..., M],
    # variables: lift.CollectionFilter = True,
    # rngs: lift.PRNGSequenceFilter = True,
    prevent_cse: bool = True,
    static_argnums: tp.Union[int, tuple[int, ...]] = (),
    policy: tp.Optional[tp.Callable[..., bool]] = None,
) -> tp.Callable[..., Remat[M]]:
    def create_remat(*args, **kwargs) -> Remat[M]:
        return Remat(
            module_constructor=module_constructor,
            module_init_args=args,
            module_init_kwargs=kwargs,
            prevent_cse=prevent_cse,
            static_argnums=static_argnums,
            policy=policy,
        )

    return create_remat


def tree_map_upto_left(
    f: tp.Callable[[tp.Any, tp.Any], tp.Any], left: tp.Any, right: tp.Any
) -> tp.Any:
    leaves_left, treedef = jtu.tree_flatten(left)
    leaves_right = treedef.flatten_up_to(right)

    return treedef.unflatten(
        f(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(leaves_left, leaves_right)
    )
