from typing import Tuple

import jax
import jax.numpy as jnp

import nnx


class Block(nnx.Module):
    def __init__(self, dim: int, *, ctx: nnx.Context):
        self.linear = nnx.Linear(dim, dim, ctx=ctx)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, ctx: nnx.Context) -> jax.Array:
        x = self.linear(x)
        x = self.dropout(x, ctx=ctx)
        x = jax.nn.gelu(x)
        return x


class ScanMLP(nnx.Module):
    """
    An MLP that uses `vmap` during `__init__` to create a Block instance
    with an additional `layer` axis, and `scan` during `__call__` to apply
    the sequence of layers iteratively over the input / output `x`.
    """

    def __init__(self, dim: int, *, n_layers: int, ctx: nnx.Context):
        self.n_layers = n_layers
        # partition Context and split the `params` key
        keys, ctxdef = ctx.partition()
        params_key = jax.random.split(keys["params"], n_layers)

        def create_block(params_key):
            # merge back Context using the sliced `params` key
            ctx = ctxdef.merge({"params": params_key})
            # create Block instance and return its partition
            return Block(dim, ctx=ctx).partition()

        # call vmap over create_block, passing the split `params` key
        # and immediately merge to get a Block instance
        self.layers = jax.vmap(create_block)(params_key).merge()

    def __call__(self, x: jax.Array, *, ctx: nnx.Context) -> jax.Array:
        # partition Context and split the `dropout` key
        keys, ctxdef = ctx.partition()
        dropout_key = jax.random.split(keys["dropout"], self.n_layers)
        # partition Module to get params
        params, moduledef = self.layers.partition("params")

        def scan_fn(
            x: jax.Array, inputs: Tuple[nnx.State, jax.Array]
        ) -> Tuple[jax.Array, nnx.State]:
            params, dropout_key = inputs
            # merge back Module and Context
            ctx = ctxdef.merge({"dropout": dropout_key})
            module = moduledef.merge(params)
            # forward pass
            x = module(x, ctx=ctx)
            # partition state and return
            params, _ = module.partition("params")
            return x, params

        # call scan passing x as the carry, and params + dropout_key as the input
        x, params = jax.lax.scan(scan_fn, x, (params, dropout_key))
        # update layers state and return
        self.layers.update_state(params)
        return x


model = ScanMLP(10, n_layers=5, ctx=nnx.context(0))

x = jnp.ones((3, 10))
y = model(x, ctx=nnx.context(dropout=1, flags=dict(deterministic=False)))

print(jax.tree_map(jnp.shape, model.get_state()))
print(y.shape)
