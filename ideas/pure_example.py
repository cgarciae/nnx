from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import pure
from pure.rngs import Rngs
from pure.state import State


@dataclass
class Linear:
  din: int
  dout: int

  def create_state(self, rngs: Rngs) -> State:
    key = rngs.make_rng("params")
    return State(
        kernel=jax.random.uniform(key, (self.din, self.dout)),
        bias=jax.numpy.zeros((self.dout,)),
    )

  def __call__(self, state: pure.State, x):
    return x @ state.kernel + state.bias


class BatchNorm(pure.Module):

  def __init__(self, din: int, mu: float = 0.95):
    self.scale = pure.Initializer(jax.random.uniform, (din,))
    self.bias = pure.Initializer(lambda _: jax.numpy.zeros((din,)))
    self.mean = pure.Initializer(
        lambda _: jax.numpy.zeros((din,)), collection="batch_stats"
    )
    self.var = pure.Initializer(
        lambda _: jax.numpy.ones((din,)), collection="batch_stats"
    )
    self.mu = mu

  def __call__(
      self, state: pure.State, x, use_running_averages: bool
  ) -> Tuple[jax.Array, pure.State]:
    scale, bias = state.scale, state.bias
    if use_running_averages:
      mean, var = state.mean, state.var
    else:
      axis = tuple(range(0, x.ndim - 1))
      mean = jax.numpy.mean(x, axis=axis)
      var = jax.numpy.var(x, axis=axis)
      # ema update
      state = state.update(
          mean=self.mu * state.mean + (1 - self.mu) * mean,
          var=self.mu * state.var + (1 - self.mu) * var,
      )

    x = (x - mean) / jax.numpy.sqrt(var + 1e-5) * scale + bias

    return x, state


class Dropout(pure.Module):

  def __init__(self, rate: float):
    raise NotImplementedError

  def __call__(self, state, rngs: Rngs, x, *, deterministic: bool) -> jax.Array:
    key = rngs.make_rng("dropout")
    raise NotImplementedError


class MLP(pure.Module):

  def __init__(self, din: int, dmid: int, dout: int):
    self.linear1 = Linear(din, dmid)
    self.bn1 = BatchNorm(dmid)
    self.dropout = Dropout(0.5)
    self.linear2 = Linear(dmid, dout)

  def __call__(
      self, state: pure.State, rngs: pure.Rngs, x: jax.Array, *, train: bool
  ) -> Tuple[jax.Array, pure.State]:
    x = self.linear1(state.linear1, x)
    x, bn1 = self.bn1(state.bn1, x, use_running_averages=not train)
    x = self.dropout(state.dropout, rngs, x, deterministic=not train)
    x = jax.nn.relu(x)
    x = self.linear2(state.linear2, x)
    return x, state.update(bn1=bn1)


model = MLP(10, 20, 30)
rngs = pure.Rngs(params=jax.random.PRNGKey(0))
state = model.create_state(rngs)


@jax.jit
def train_step(state: pure.State, key, batch):
  x, y = batch
  params = state.partition(nnx.Param)
  rngs = pure.Rngs(dropout=key)

  def loss(params):
    _state = state.merge(params)
    y_pred, _state = model(_state, rngs, x, train=True)
    loss = jax.numpy.mean((y_pred - y) ** 2)
    return loss, _state

  grads, state = jax.grad(loss, has_aux=True)(params)
  params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)
  state = state.merge(params)

  return state


# ----------------------------------------
# scan over layers + shared batch_stats
# ----------------------------------------

model = MLP(10, 20, 10)
n_layers = 10
params_keys = jax.random.PRNGKey(0)
params_keys = jax.random.split(params_keys, n_layers)


@partial(jax.vmap, in_axes=0, out_axes=(0, None))
def create_state(params_key: jax.random.KeyArray):
  state = model.create_state(pure.Rngs(params=params_key))
  params, batch_stats = state.partition(nnx.Param, "batch_stats")
  return params, batch_stats


params, batch_stats = create_state(params_keys)
x = jax.numpy.zeros((32, 10))
dropout_key = jax.random.PRNGKey(1)
dropout_stream = pure.RngStream(jax.random.split(dropout_key, n_layers))


def scan_fn(
    carry: Tuple[jax.Array, pure.Partition],
    inputs: Tuple[pure.Partition, pure.RngStream],
):
  # extract args
  x, batch_stats = carry
  params, dropout_stream = inputs

  # create state and rngs
  state = pure.merge(params, batch_stats)
  rngs = pure.Rngs(dropout=dropout_stream)

  # forward pass
  x, state = model(state, rngs, x, train=True)

  # partition state
  params, batch_stats = state.partition(nnx.Param, "batch_stats")

  return (x, batch_stats), params


(y, batch_stats), params = jax.lax.scan(
    scan_fn, (x, batch_stats), (params, dropout_stream)
)
state = pure.merge(params, batch_stats)
