# %%
from typing import Any
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import nnx

X = np.linspace(0, 1, 100)[:, None]
Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
    while True:
        idx = np.random.choice(len(X), size=batch_size)
        yield X[idx], Y[idx]


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        self.w = nnx.param(jax.random.uniform(ctx.make_rng("params"), (din, dout)))
        self.b = nnx.param(jnp.zeros((dout,)))

    def __call__(self, x):
        return jnp.dot(x, self.w.value) + self.b.value


class MLP(nnx.Module):
    def __init__(self, din, dhidden, dout, *, ctx: nnx.Context):
        self.count = nnx.ref("state", jnp.array(0))
        self.linear1 = Linear(din, dhidden, ctx=ctx)
        self.linear2 = Linear(dhidden, dout, ctx=ctx)

    def __call__(self, x):
        self.count.value += 1
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


@jax.jit
def train_step(derefmod: nnx.DerefedMod[Any, MLP], batch) -> nnx.DerefedMod[Any, MLP]:
    x, y = batch
    model = derefmod.reref()

    def loss_fn(model: MLP):
        y_pred = model(x)
        loss = jnp.mean((y - y_pred) ** 2)
        return loss

    grads = nnx.grad(loss_fn)(model)
    #                              |-------- sgd ---------|
    model.update[:] = jax.tree_map(lambda w, g: w - 0.1 * g, model.get("params"), grads)

    return model.deref()


@jax.jit
def test_step(unbound: nnx.DerefedMod[Any, MLP], batch):
    x, y = batch
    model = unbound.reref()
    y_pred = model(x)
    loss = jnp.mean((y - y_pred) ** 2)
    return {"loss": loss}


ctx = nnx.Context(jax.random.PRNGKey(0))
model = MLP(din=1, dhidden=32, dout=1, ctx=ctx)
derefmod = model.deref()


total_steps = 10_000
for step, batch in enumerate(dataset(32)):
    derefmod = train_step(derefmod, batch)

    if step % 1000 == 0:
        logs = test_step(derefmod, (X, Y))
        print(f"step: {step}, loss: {logs['loss']}")

    if step >= total_steps - 1:
        break

model = derefmod.reref()
print("times called:", model.count.value)

y_pred = model(X)

plt.scatter(X, Y, color="blue")
plt.plot(X, y_pred, color="black")
plt.show()
