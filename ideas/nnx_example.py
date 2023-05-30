from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import nnx


class Linear(nnx.Module):
    kernel: jax.Array = nnx.param()
    bias: jax.Array = nnx.param()

    def __init__(self, din: int, dout: int):
        self.kernel = jax.random.uniform(nnx.make_rng("params"), (din, dout))
        self.bias = jax.numpy.zeros((dout,))

    def __call__(self, x):
        return x @ self.kernel + self.bias


class BatchNorm(nnx.Module):
    scale: jax.Array = nnx.param()
    bias: jax.Array = nnx.param()
    mean: jax.Array = nnx.ref("batch_stats")
    var: jax.Array = nnx.ref("batch_stats")
    mu: float = nnx.static_field()

    def __init__(self, din: int, mu: float = 0.95):
        self.scale = jax.random.uniform(nnx.make_rng("params"), (din,))
        self.bias = jax.numpy.zeros((din,))
        self.mean = jax.numpy.zeros((din,))
        self.var = jax.numpy.ones((din,))
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


@nnx.dataclasses
class Dropout(nnx.Module):
    rate: float

    def __call__(self, inputs, *, deterministic: bool):
        if (self.rate == 0.0) or deterministic:
            return inputs
        rng = nnx.make_rng("dropout")
        keep_prob = 1.0 - self.rate
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=inputs.shape)
        return jax.lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        self.linear1 = Linear(din, dmid)
        self.bn1 = BatchNorm(dmid)
        self.dropout = Dropout(0.5)
        self.linear2 = Linear(dmid, dout)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = self.linear1(x)
        x = self.bn1(x, use_running_averages=not train)
        x = self.dropout(x, deterministic=not train)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


rngs = nnx.Context(jax.random.PRNGKey(0))
model = MLP.init(rngs)(10, 20, 30)


@nnx.jit
def train_step(model: MLP, key, batch):
    x, y = batch

    def loss(model: MLP):
        rngs = nnx.Context(dropout=key)
        y_pred = model.apply(rngs=rngs)(x, train=True)
        loss = jax.numpy.mean((y_pred - y) ** 2)
        return loss

    grads = nnx.grad(loss, wrt="params")(model)
    model[:] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grads)


# ----------------------------------------
# scan over layers + shared batchnorm
# ----------------------------------------

n_layers = 10
params_keys = jax.random.PRNGKey(0)
params_keys = jax.random.split(params_keys, n_layers)


@partial(jax.vmap, in_axes=0, out_axes=(0, None, None))
def create_state(params_key: jax.random.KeyArray):
    rngs = nnx.Context(params=params_key)
    model = MLP.init(rngs)(10, 20, 10)
    (params, batch_stats), modeldef = model.partition("params", "batch_stats")
    return params, batch_stats, modeldef


params, batch_stats, modeldef = create_state(params_keys)
x = jax.numpy.zeros((32, 10))
dropout_key = jax.random.PRNGKey(1)
dropout_stream = nnx.RngStream(jax.random.split(dropout_key, n_layers))


def scan_fn(
    carry: Tuple[jax.Array, nnx.State],
    inputs: Tuple[nnx.State, nnx.RngStream],
):
    # extract args
    x, batch_stats = carry
    params, dropout_stream = inputs

    # create state and rngs
    model = modeldef.merge([params, batch_stats])
    rngs = nnx.Context(dropout=dropout_stream)

    # forward pass
    x = model.apply(rngs=rngs)(x, train=True)

    # partition state
    params, batch_stats = model.partition("params", "batch_stats")[0]

    return (x, batch_stats), params


(y, batch_stats), params = jax.lax.scan(
    scan_fn, (x, batch_stats), (params, dropout_stream)
)
model = modeldef.merge([params, batch_stats])
