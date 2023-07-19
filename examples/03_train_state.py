# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training import train_state

import nnx

X = np.linspace(0, 1, 100)[:, None]
Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
  while True:
    idx = np.random.choice(len(X), size=batch_size)
    yield X[idx], Y[idx]


class Linear(nnx.Module):

  def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
    self.w = nnx.Param(jax.random.uniform(ctx.make_rng("params"), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))

  def __call__(self, x):
    return x @ self.w + self.b


class MLP(nnx.Module):

  def __init__(self, din, dhidden, dout, *, ctx: nnx.Context):
    self.count = jnp.array(0)
    self.linear1 = Linear(din, dhidden, ctx=ctx)
    self.linear2 = Linear(dhidden, dout, ctx=ctx)

  def __call__(self, x):
    self.count += 1
    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    return x


(params, buffers), moduledef = MLP(
    din=1, dhidden=32, dout=1, ctx=nnx.context(0)
).partition(nnx.Param, ...)

state = nnx.TrainState(
    moduledef,
    params=params,
    tx=optax.sgd(0.1),
    buffers=buffers,
)
del params, buffers


@jax.jit
def train_step(state: nnx.TrainState, batch):
  x, y = batch

  def loss_fn(params):
    y_pred, (updates, _) = state.apply(params, "buffers")(x)
    buffers = updates.filter(nnx.buffers)
    loss = jnp.mean((y - y_pred) ** 2)
    return loss, buffers

  grads, buffers = jax.grad(loss_fn, has_aux=True)(state.params)
  # sdg update
  state = state.apply_gradients(grads=grads, buffers=buffers)

  return state


@jax.jit
def test_step(state: nnx.TrainState, batch):
  x, y = batch
  y_pred, _ = state.apply("params", "buffers")(x)
  loss = jnp.mean((y - y_pred) ** 2)
  return {"loss": loss}


total_steps = 10_000
for step, batch in enumerate(dataset(32)):
  state = train_step(state, batch)

  if step % 1000 == 0:
    logs = test_step(state, (X, Y))
    print(f"step: {step}, loss: {logs['loss']}")

  if step >= total_steps - 1:
    break

model = moduledef.merge(state.params, state.buffers)
print("times called:", model.count)

y_pred = model(X)

plt.scatter(X, Y, color="blue")
plt.plot(X, y_pred, color="black")
plt.show()
