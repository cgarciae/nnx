# %%
import ciclo
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import simple_pytree as spt

import nnx
import refx

X = np.linspace(0, 1, 100)[:, None]
Y = 0.8 * X + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
    while True:
        idx = np.random.choice(len(X), size=batch_size)
        yield X[idx], Y[idx]


class LinearClassifier(spt.Pytree):
    def __init__(self, din: int, dout: int, *, key: jax.random.KeyArray):
        self.w = nnx.Param(jax.nn.initializers.kaiming_normal()(key, (dout, din)))
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x):
        return jnp.dot(x, self.w.value) + self.b.value


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
    params = refx.get_partition(refx.deref(model), nnx.Param)
    params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grad)
    refx.update_partition(model, params, nnx.Param)

    logs = ciclo.logs()
    logs.add_metric("loss", loss)

    return logs, model


@nnx.jit
def test_step(model: LinearClassifier, batch):
    y_pred = model(X)
    loss = mse(Y, y_pred)

    logs = ciclo.logs()
    logs.add_metric("mse", loss)

    return logs


total_steps = 10_000

model = LinearClassifier(din=1, dout=1, key=jax.random.PRNGKey(0))
model, history, elapsed = ciclo.loop(
    state=model,
    dataset=dataset(batch_size=32),
    tasks={
        ciclo.every(100): [test_step],
        ciclo.every(1): [
            train_step,
            ciclo.keras_bar(total=total_steps),
        ],
    },
    stop=total_steps,
)


y_pred = model(X)

plt.scatter(X, Y, color="blue")
plt.plot(X, y_pred, color="black")
plt.show()

# %%
