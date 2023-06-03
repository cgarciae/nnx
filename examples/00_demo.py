# shared leafs   + explicit refs
# ------------ vs --------------
# shared Modules + implicit vars

import jax
import jax.numpy as jnp
import optax
import nnx


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        self.w = nnx.param(jax.random.uniform(ctx.make_rng("params"), (din, dout)))
        self.b = nnx.param(jnp.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w + self.b


ctx = nnx.Context(jax.random.PRNGKey(0))
linear = Linear(2, 2, ctx=ctx)


state, moduledef = linear.split()
params, batch_stats = state.split("params", "batch_stats")
linear2 = moduledef.merge(params, batch_stats)
