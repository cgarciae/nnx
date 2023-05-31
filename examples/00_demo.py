# shared leafs   + explicit refs
# ------------ vs --------------
# shared Modules + implicit vars

import jax
import jax.numpy as jnp
import nnx


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        self.w = nnx.param(jax.random.uniform(ctx.make_rng("params"), (din, dout)))
        self.b = nnx.param(jnp.zeros((dout,)))
        self.list = nnx.Seq([nnx.Conv(10, 20, [3, 3], ctx=ctx), 2, 3])

    def __call__(self, x):
        self.interm = nnx.ref("interm", 100)
        return x @ self.w.value + self.b.value


ctx = nnx.Context(jax.random.PRNGKey(0))
linear = Linear(2, 2, ctx=ctx)

state, moddef = statedef = linear.deref()

print(state)

# weight tying
linear2 = Linear(2, 2, ctx=ctx)
linear2.w = linear.w
