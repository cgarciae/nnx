from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import pure_pytree as pure


class Linear(pure.Module):
    kernel: jax.Array = pure.param()
    bias: jax.Array = pure.param()

    def __init__(self, din: int, dout: int, *, rngs: pure.Rngs):
        self.kernel = jax.random.uniform(rngs.make_rng("params"), (din, dout))
        self.bias = jax.numpy.zeros((dout,))

    def __call__(self, x):
        return x @ self.kernel + self.bias


class BatchNorm(pure.Module):
    scale: jax.Array = pure.param()
    bias: jax.Array = pure.param()
    mean: jax.Array = pure.variable("batch_stats")
    var: jax.Array = pure.variable("batch_stats")
    mu: float = pure.static_field()

    def __init__(self, din: int, mu: float = 0.95, *, rngs: pure.Rngs):
        self.scale = jax.random.uniform(rngs.make_rng("params"), (din,))
        self.bias = jax.numpy.zeros((din,))
        self.mean = jax.numpy.zeros((din,))
        self.var = jax.numpy.ones((din,))
        self.mu = mu

    def __call__(self, x, use_running_averages: bool) -> Tuple[jax.Array, "BatchNorm"]:
        scale, bias = self.scale, self.bias
        if use_running_averages:
            mean, var = self.mean, self.var
        else:
            axis = tuple(range(0, x.ndim - 1))
            mean = jax.numpy.mean(x, axis=axis)
            var = jax.numpy.var(x, axis=axis)
            # ema update
            self = self.replace(
                mean=self.mu * self.mean + (1 - self.mu) * mean,
                var=self.mu * self.var + (1 - self.mu) * var,
            )

        x = (x - mean) / jax.numpy.sqrt(var + 1e-5) * scale + bias

        return x, self


class Dropout(pure.Module):
    def __init__(self, rate: float):
        raise NotImplementedError

    def __call__(self, x, *, deterministic: bool, rngs: pure.Rngs) -> jax.Array:
        key = rngs.make_rng("dropout")
        ...
        raise NotImplementedError


class MLP(pure.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: pure.Rngs):
        self.linear1 = Linear(din, dmid, rngs=rngs)
        self.bn1 = BatchNorm(dmid, rngs=rngs)
        self.dropout = Dropout(0.5)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

    def __call__(
        self, x: jax.Array, *, train: bool, rngs: pure.Rngs
    ) -> Tuple[jax.Array, "MLP"]:
        x = self.linear1(x)
        x, bn1 = self.bn1(x, use_running_averages=not train)
        x = self.dropout(x, deterministic=not train, rngs=rngs)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x, self.replace(bn1=bn1)


rngs = pure.Rngs(params=jax.random.PRNGKey(0))
model = MLP(10, 20, 30, rngs=rngs)


@jax.jit
def train_step(model: MLP, key, batch):
    x, y = batch
    params = model.get_partition("params")
    rngs = pure.Rngs(dropout=key)

    def loss(params):
        _model = model.merge(params)
        y_pred, _model = model(x, train=True, rngs=rngs)
        loss = jax.numpy.mean((y_pred - y) ** 2)
        return loss, _model

    grads, model = jax.grad(loss, has_aux=True)(params)
    params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)
    model = model.merge(params)

    return model


# ----------------------------------------
# scan over layers + shared batchnorm
# ----------------------------------------

n_layers = 10
params_keys = jax.random.PRNGKey(0)
params_keys = jax.random.split(params_keys, n_layers)


@partial(jax.vmap, in_axes=0, out_axes=(0, None, None))
def create_state(params_key: jax.random.KeyArray):
    rngs = pure.Rngs(params=params_key)
    model = MLP(10, 20, 10, rngs=rngs)
    (params, batch_stats), modeldef = model.partition(nnx.Param, "batch_stats")
    return params, batch_stats, modeldef


params, batch_stats, modeldef = create_state(params_keys)
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
    model = pure.merge([params, batch_stats], modeldef)
    rngs = pure.Rngs(dropout=dropout_stream)

    # forward pass
    x, model = model(x, train=True, rngs=rngs)

    # partition state
    params, batch_stats = model.get_partition("params", "batch_stats")

    return (x, batch_stats), params


(y, batch_stats), params = jax.lax.scan(
    scan_fn, (x, batch_stats), (params, dropout_stream)
)
model = pure.merge([params, batch_stats], modeldef)
