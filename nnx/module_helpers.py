import typing as tp

import jax.numpy as jnp
import optax

import nnx
from nnx.module import ApplyCaller, Module, ModuleDef
from nnx.state import State

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound=Module)


class Map(Module, tp.Mapping[str, A]):
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


class Seq(Module, tp.Generic[A]):
    def __init__(self, iterable: tp.Iterable[A]):
        for i, value in enumerate(iterable):
            setattr(self, str(i), value)

    def __getitem__(self, key: int) -> A:
        if key >= len(self):
            raise IndexError(f"index {key} out of range for {self}")
        return getattr(self, str(key))

    def __iter__(self) -> tp.Iterator[A]:
        for i in range(len(self)):
            yield getattr(self, str(i))

    def __len__(self) -> int:
        return len(vars(self))


class TrainState(Module):
    def __init__(
        self,
        *,
        apply_fn: tp.Callable[
            [tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]]], ApplyCaller
        ],
        params: State,
        tx: optax.GradientTransformation,
        step: int = 0,
        **kwargs,
    ):
        self.apply_fn = apply_fn
        self.params: nnx.State = params
        self.tx = tx
        self.opt_state = nnx.var("opt_state", tx.init(self.params))
        self.step = jnp.asarray(step)
        for name, value in kwargs.items():
            setattr(self, name, value)

    if tp.TYPE_CHECKING:

        def __getattr__(self, key: str) -> tp.Any:
            ...

    def apply_gradients(self, grads: nnx.State):
        updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)  # type: ignore
        self.step += 1
