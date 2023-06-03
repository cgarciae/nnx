# %%
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
        return x @ self.w + self.b


class MLP(nnx.Module):
    def __init__(self, din, dhidden, dout, *, ctx: nnx.Context):
        self.count = nnx.var("state", jnp.array(0))
        self.linear1 = Linear(din, dhidden, ctx=ctx)
        self.linear2 = Linear(dhidden, dout, ctx=ctx)

    def __call__(self, x):
        self.count += 1
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


def mse(y, y_pred):
    return jnp.mean((y - y_pred) ** 2)


@nnx.jit
def train_step(model: MLP, batch):
    x, y = batch

    def loss_fn(model: MLP):
        y_pred = model(x)
        return jnp.mean((y - y_pred) ** 2)

    #                                      |--default--|
    grad: nnx.State = nnx.grad(loss_fn, wrt="params")(model)
    #                         |-------- sgd ---------|
    model.update(
        jax.tree_map(lambda w, g: w - 0.1 * g, model.get_state("params"), grad)
    )

    # no return!!!


@nnx.jit
def test_step(model: MLP, batch):
    x, y = batch
    y_pred = model(x)
    loss = jnp.mean((y - y_pred) ** 2)
    return {"loss": loss}


ctx = nnx.Context(jax.random.PRNGKey(0))
model = MLP(din=1, dhidden=32, dout=1, ctx=ctx)

total_steps = 10_000
for step, batch in enumerate(dataset(32)):
    train_step(model, batch)

    if step % 1000 == 0:
        logs = test_step(model, (X, Y))
        print(f"step: {step}, loss: {logs['loss']}")

    if step >= total_steps - 1:
        break

print("times called:", model.count)

y_pred = model(X)

plt.scatter(X, Y, color="blue")
plt.plot(X, y_pred, color="black")
plt.show()
