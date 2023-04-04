# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.nn import initializers

import nnx

X = np.linspace(0, 1, 100)[:, None]
Y = 0.8 * X + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
    while True:
        idx = np.random.choice(len(X), size=batch_size)
        yield X[idx], Y[idx]


class LinearClassifier(nnx.Module):
    w: jax.Array = nnx.param()
    b: jax.Array = nnx.param()
    count: jax.Array = nnx.reference("state")

    def __init__(
        self,
        din: int,
        dout: int,
        kernel_init: initializers.Initializer = initializers.kaiming_normal(),
    ):
        w_key = nnx.make_rng("params")
        self.w = kernel_init(w_key, (dout, din))
        self.b = jnp.zeros((dout,))
        self.count = jnp.array(0)

    def __call__(self, x):
        self.count += 1
        return jnp.dot(x, self.w) + self.b


def mse(y, y_pred):
    return jnp.mean((y - y_pred) ** 2)


@nnx.jit
def train_step(model: LinearClassifier, batch):
    x, y = batch

    def loss_fn(model: LinearClassifier):
        y_pred = model(x)
        return mse(y, y_pred)

    grad_fn = nnx.grad(loss_fn)
    loss = loss_fn(model)
    grad = grad_fn(model)

    # sdg
    model["params"] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)

    return {"loss": loss}


@nnx.jit
def test_step(model: LinearClassifier, batch):
    x, y = batch
    y_pred = model(x)
    loss = mse(y, y_pred)

    return {"loss": loss}


total_steps = 10_000

model = LinearClassifier.init(jax.random.PRNGKey(0))(din=1, dout=1)

for step, batch in enumerate(dataset(32)):
    train_step(model, batch)

    if step % 100 == 0:
        logs = test_step(model, (X, Y))
        print(f"step: {step}, loss: {logs['loss']}")

    if step >= total_steps:
        break

y_pred = model(X)

print("step:", model.count)

plt.scatter(X, Y, color="blue")
plt.plot(X, y_pred, color="black")
plt.show()

# %%
