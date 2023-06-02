# shared leafs   + explicit refs
# ------------ vs --------------
# shared Modules + implicit vars

import jax
import jax.numpy as jnp
import optax
import nnx
from flax.training.train_state import TrainState


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        self.w = nnx.param(jax.random.uniform(ctx.make_rng("params"), (din, dout)))
        self.b = nnx.param(jnp.zeros((dout,)))

    def __call__(self, x):
        self.interm = nnx.var("interm", 100)
        return x @ self.w + self.b


ctx = nnx.Context(jax.random.PRNGKey(0))
linear = Linear(2, 2, ctx=ctx)


# split / merge
y = linear(jnp.ones((2, 2)))
(params, interm), moduledef = linear.split("params", "interm")
linear = moduledef.merge(params)

# pop
y = linear(jnp.ones((2, 2)))
interm = linear.pop("interm")
