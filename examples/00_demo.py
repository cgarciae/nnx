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
        self.list = nnx.Seq([nnx.Conv(10, 20, [3, 3], ctx=ctx), 2, 3])

    def __call__(self, x):
        self.interm = nnx.var("interm", 100)
        return x @ self.w + self.b


ctx = nnx.Context(jax.random.PRNGKey(0))
linear = Linear(2, 2, ctx=ctx)

y = linear(jnp.ones((2, 2)))


params, moddef = linear.split("params")
state = TrainState.create(
    apply_fn=moddef.apply,
    params=params,
    tx=optax.adam(1e-3),
)

y, updates = state.apply_fn(state.params)(x)

model = moddef.merge((params, batch_stats))

linear.update(params)
# print(state)
