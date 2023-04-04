# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.nn import initializers

import nnx

X = np.linspace(0, 1, 100)[:, None]
Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
    while True:
        idx = np.random.choice(len(X), size=batch_size)
        yield X[idx], Y[idx]


class Linear(nnx.Module):
    w: jax.Array = nnx.param()
    b: jax.Array = nnx.param()

    def __init__(
        self,
        din: int,
        dout: int,
        kernel_init: initializers.Initializer = initializers.kaiming_normal(),
    ):
        w_key = nnx.make_rng("params")
        self.w = kernel_init(w_key, (din, dout))
        self.b = jnp.zeros((dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


class MLP(nnx.Module):
    count: jax.Array = nnx.reference("state")

    def __init__(self, din, dhidden, dout):
        self.count = jnp.array(0)
        self.linear1 = Linear(din, dhidden)
        self.linear2 = Linear(dhidden, dout)

    def __call__(self, x):
        self.count += 1
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


def mse(y, y_pred):
    return jnp.mean((y - y_pred) ** 2)


@jax.jit
def train_step(model: nnx.ModuleDef[MLP], params, state, batch):
    x, y = batch

    def loss_fn(params):
        y_pred, updates = model.apply([params, state])(x)
        loss = mse(y, y_pred)
        return loss, updates["state"]

    grad, state = jax.grad(loss_fn, has_aux=True)(params)
    #                          |-------- sgd ---------|
    params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grad)

    return params, state


@jax.jit
def test_step(model: nnx.ModuleDef[MLP], params, state, batch):
    x, y = batch
    y_pred, _ = model.apply([params, state])(x)
    loss = mse(y, y_pred)
    return {"loss": loss}


total_steps = 10_000

model = MLP.init(jax.random.PRNGKey(0))(din=1, dhidden=32, dout=1)
(params, state), model = model.partition("params", "state")


for step, batch in enumerate(dataset(32)):
    params, state = train_step(model, params, state, batch)

    if step % 100 == 0:
        logs = test_step(model, params, state, (X, Y))
        print(f"step: {step}, loss: {logs['loss']}")

    if step >= total_steps - 1:
        break

model = model.merge([params, state])
print("times called:", model.count)

y_pred = model(X)

plt.scatter(X, Y, color="blue")
plt.plot(X, y_pred, color="black")
plt.show()
