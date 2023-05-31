import typing as tp

import jax.numpy as jnp
import optax

from nnx.module import Module
import nnx

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound=Module)


class Map(Module, tp.Mapping[str, A]):
    def __init__(self, *args, **kwargs):
        vars(self).update(*args, **kwargs)

    def __getitem__(self, key) -> A:
        return vars(self)[key]

    def __iter__(self) -> tp.Iterator[str]:
        return iter(vars(self))

    def __len__(self) -> int:
        return len(vars(self))


class Seq(Module, tp.Generic[A]):
    def __init__(self, iterable: tp.Iterable[A]):
        for i, value in enumerate(iterable):
            vars(self)[str(i)] = value

    def __getitem__(self, key: int) -> A:
        if key >= len(self):
            raise IndexError(f"Index {key} out of range for {self}")
        return vars(self)[str(key)]

    def __iter__(self) -> tp.Iterator[A]:
        for i in range(len(self)):
            yield vars(self)[str(i)]

    def __len__(self) -> int:
        return len(vars(self))


from flax.training import train_state


class TrainState(Module, tp.Generic[M]):
    def __init__(self, model: M, tx: optax.GradientTransformation, *, step: int = 0):
        self.model = model
        self.tx = tx
        self.opt_state = nnx.ref("opt_state", tx.init(model.get("params")))
        self.step = jnp.asarray(step)

    def apply_gradients(self, grads: nnx.State):
        params: nnx.State = self.model.get("params")
        updates, self.opt_state.value = self.tx.update(
            grads, self.opt_state.value, params
        )
        params = optax.apply_updates(params, updates)  # type: ignore
        self.model.update(params)
        self.step += 1
