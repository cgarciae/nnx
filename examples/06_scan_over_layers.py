import dataclasses
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

import nnx


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        self.kernel = nnx.param(jax.random.uniform(ctx.make_rng("params"), (din, dout)))
        self.bias = nnx.param(jax.numpy.zeros((dout,)))

    def __call__(self, x):
        return x @ self.kernel + self.bias


class BatchNorm(nnx.Module):
    def __init__(self, din: int, mu: float = 0.95, *, ctx: nnx.Context):
        self.scale = nnx.param(jax.random.uniform(ctx.make_rng("params"), (din,)))
        self.bias = nnx.param(jax.numpy.zeros((din,)))
        self.mean = nnx.var("batch_stats", jax.numpy.zeros((din,)))
        self.var = nnx.var("batch_stats", jax.numpy.ones((din,)))
        self.mu = mu

    def __call__(self, x, *, use_running_averages: bool) -> jax.Array:
        scale, bias = self.scale, self.bias
        if use_running_averages:
            mean, var = self.mean, self.var
        else:
            axis = tuple(range(0, x.ndim - 1))
            mean = jax.numpy.mean(x, axis=axis)
            var = jax.numpy.var(x, axis=axis)
            # ema update
            self.mean = self.mu * self.mean + (1 - self.mu) * mean
            self.var = self.mu * self.var + (1 - self.mu) * var

        x = (x - mean) / jax.numpy.sqrt(var + 1e-5) * scale + bias

        return x


@dataclasses.dataclass
class Dropout(nnx.Module):
    rate: float

    def __call__(self, inputs, *, deterministic: bool, ctx: nnx.Context):
        if (self.rate == 0.0) or deterministic:
            return inputs
        rng = ctx.make_rng("dropout")
        keep_prob = 1.0 - self.rate
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=inputs.shape)
        return jax.lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, ctx: nnx.Context):
        self.linear1 = Linear(din, dmid, ctx=ctx)
        self.bn1 = BatchNorm(dmid, ctx=ctx)
        self.dropout = Dropout(0.5)
        self.linear2 = Linear(dmid, dout, ctx=ctx)

    def __call__(self, x: jax.Array, *, train: bool, ctx: nnx.Context) -> jax.Array:
        x = self.linear1(x)
        x = self.bn1(x, use_running_averages=not train)
        x = self.dropout(x, deterministic=not train, ctx=ctx)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


model = MLP(10, 20, 30, ctx=nnx.context(0))


@nnx.jit
def train_step(model: MLP, key, batch):
    x, y = batch

    def loss(model: MLP):
        y_pred = model(x, train=True, ctx=nnx.context(dropout=key))
        loss = jax.numpy.mean((y_pred - y) ** 2)
        return loss

    grads = nnx.grad(loss, wrt="params")(model)
    model.update_state(
        jax.tree_map(lambda w, g: w - 0.1 * g, model.filter("params"), grads)
    )


# ----------------------------------------
# scan over layers + shared batchnorm
# ----------------------------------------

n_layers = 10
params_keys = jax.random.PRNGKey(0)
params_keys = jax.random.split(params_keys, n_layers)


@partial(jax.vmap, in_axes=0, out_axes=(0, None, None))
def create_state(params_key: jax.random.KeyArray):
    model = MLP(10, 20, 10, ctx=nnx.context(params_key))
    (params, batch_stats), modeldef = model.partition("params", "batch_stats")
    return params, batch_stats, modeldef


params, batch_stats, modeldef = create_state(params_keys)
x = jax.numpy.zeros((32, 10))
dropout_key = jax.random.split(jax.random.PRNGKey(1), n_layers)


def scan_fn(
    carry: Tuple[jax.Array, nnx.State],
    inputs: Tuple[nnx.State, jax.random.KeyArray],
):
    # extract args
    x, batch_stats = carry
    params, dropout_key = inputs

    # create state and ctx
    model = modeldef.merge(params, batch_stats)
    ctx = nnx.context(dropout=dropout_key)

    # forward pass
    x = model(x, train=True, ctx=ctx)

    # partition state
    (params, batch_stats), _ = model.partition("params", "batch_stats")

    return (x, batch_stats), params


(y, batch_stats), params = jax.lax.scan(
    scan_fn, (x, batch_stats), (params, dropout_key)
)
model = modeldef.merge(params, batch_stats)
