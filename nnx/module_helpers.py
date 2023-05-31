import typing as tp

import jax.numpy as jnp
import optax

import nnx
from nnx.module import Module

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


class TrainState(Module, tp.Generic[M]):
    def __init__(self, model: M, tx: optax.GradientTransformation, *, step: int = 0):
        self.model = model
        self.tx = tx
        self.opt_state = nnx.var("opt_state", tx.init(model.get("params")))
        self.step = jnp.asarray(step)

    def apply_gradients(self, grads: nnx.State):
        params: nnx.State = self.model.get("params")
        updates, self.opt_state = self.tx.update(grads, self.opt_state, params)
        params = optax.apply_updates(params, updates)  # type: ignore
        self.model.update(params)
        self.step += 1
