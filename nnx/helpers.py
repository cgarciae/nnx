import inspect
import typing as tp

import jax.numpy as jnp
import optax

from nnx import pytreelib
from nnx.contextlib import Context
from nnx.module import ApplyCaller, Module, ModuleDef, Pure
from nnx.state import State

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound=Module)


class Dict(Module, tp.Mapping[str, A]):
    @tp.overload
    def __init__(self, __iterable: tp.Iterable[tp.Tuple[str, A]]):
        ...

    @tp.overload
    def __init__(self, __mapping: tp.Optional[tp.Mapping[str, A]] = None, **kwargs: A):
        ...

    def __init__(self, *args, **kwargs):
        for name, value in dict(*args, **kwargs).items():
            setattr(self, name, value)

    def __getitem__(self, key) -> A:
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getattr__(self, key) -> A:
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    def __iter__(self) -> tp.Iterator[str]:
        return iter(vars(self))

    def __len__(self) -> int:
        return len(vars(self))


class Sequence(Module, tp.Generic[A]):
    def __init__(self, iterable: tp.Iterable[A]):
        i = 0
        for i, value in enumerate(iterable):
            setattr(self, str(i), value)
        self._length = i + 1

    def __getitem__(self, key: int) -> A:
        if key >= len(self):
            raise IndexError(f"index {key} out of range for {self}")
        return getattr(self, str(key))

    def __iter__(self) -> tp.Iterator[A]:
        for i in range(len(self)):
            yield getattr(self, str(i))

    def __len__(self) -> int:
        return self._length

    def __call__(self, *args, ctx: tp.Optional[Context] = None, **kwargs) -> tp.Any:
        output: tp.Any = None

        for i, f in enumerate(self):
            if not callable(f):
                raise TypeError(f"Sequence[{i}] is not callable: {f}")
            if i > 0:
                if isinstance(output, tp.Tuple):
                    args = output
                    kwargs = {}
                elif isinstance(output, tp.Dict):
                    args = ()
                    kwargs = output
                else:
                    args = (output,)
                    kwargs = {}
            if ctx is not None and has_keyword_arg(f, "ctx"):
                kwargs["ctx"] = ctx

            output = f(*args, **kwargs)

        return output


class ModuleDefApply(tp.Protocol, tp.Generic[M]):
    def __call__(self, state: State, *states: State) -> ApplyCaller["Pure[M]"]:
        ...


class TrainState(pytreelib.Pytree, tp.Generic[M]):
    def __init__(
        self,
        moduledef: ModuleDef[M],
        *,
        params: State,
        tx: optax.GradientTransformation,
        step: int = 0,
        **kwargs,
    ):
        self.moduledef = moduledef
        self.params: State = pytreelib.tree_node(params)
        self.tx = tx
        self.opt_state = pytreelib.tree_node(tx.init(self.params))
        self.step = pytreelib.tree_node(jnp.asarray(step))
        for name, value in kwargs.items():
            setattr(self, name, value)

    if tp.TYPE_CHECKING:

        def __getattr__(self, key: str) -> tp.Any:
            ...

    def apply(
        self, state: tp.Union[State, str], *states: tp.Union[State, str]
    ) -> ApplyCaller[Pure[State, M]]:
        states = (state, *states)

        _states = (
            getattr(self, state) if isinstance(state, str) else state
            for state in states
        )

        return self.moduledef.apply(*_states)

    def apply_gradients(self, grads: State, **kwargs) -> "TrainState[M]":
        updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)  # type: ignore
        step = self.step + 1
        return self.replace(
            params=params,
            opt_state=opt_state,
            step=step,
            **kwargs,
        )


def has_keyword_arg(func: tp.Callable[..., tp.Any], name: str) -> bool:
    """Return True if func has keyword-only arguments with the given name."""
    return any(
        param.name == name
        and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
        for param in inspect.signature(func).parameters.values()
    )
