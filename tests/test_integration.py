import jax
import jax.numpy as jnp
import numpy as np

import nnx


class TestIntegration:
    def test_shared_modules(self):
        class Block(nnx.Module):
            def __init__(self, linear: nnx.Linear, *, ctx):
                self.linear = linear
                self.bn = nnx.BatchNorm(2, ctx=ctx)

            def __call__(self, x, *, ctx):
                x = self.linear(x)
                x = self.bn(x, ctx=ctx)
                return nnx.relu(x)

        class Model(nnx.Module):
            def __init__(self, *, ctx):
                shared = nnx.Linear(2, 2, ctx=ctx)
                self.block1 = Block(shared, ctx=ctx)
                self.block2 = Block(shared, ctx=ctx)

            def __call__(self, x, *, ctx):
                x = self.block1(x, ctx=ctx)
                x = self.block2(x, ctx=ctx)
                return x

        @nnx.jit
        def train_step(model: Model, x, y):
            @nnx.grad
            def loss_fn(model: Model):
                ctx = nnx.Context(flags=dict(use_running_average=False))
                y_pred = model(x, ctx=ctx)
                return jnp.mean((y - y_pred) ** 2)

            grads = loss_fn(model)
            model.update[:] = jax.tree_map(
                lambda w, g: w - 0.1 * g, model.get("params"), grads
            )

        ctx = nnx.Context(jax.random.PRNGKey(0))
        model = Model(ctx=ctx)

        x = np.random.uniform(size=(4, 2))
        y = np.random.uniform(size=(4, 2))

        for _i in range(3):
            train_step(model, x, y)

        assert model.block1.linear is model.block2.linear
        assert model.block1.linear.bias is not None
        assert model.block1.bn is not model.block2.bn

    def test_shared_modules_jit_filter(self):
        class Block(nnx.Module):
            def __init__(self, linear: nnx.Linear, *, ctx: nnx.Context):
                self.linear = linear
                self.bn = nnx.BatchNorm(2, ctx=ctx)

            def __call__(self, x, *, ctx: nnx.Context):
                x = self.linear(x)
                x = self.bn(x, ctx=ctx)
                return nnx.relu(x)

        class Model(nnx.Module):
            def __init__(self, *, ctx: nnx.Context):
                shared = nnx.Linear(2, 2, ctx=ctx)
                self.block1 = Block(shared, ctx=ctx)
                self.block2 = Block(shared, ctx=ctx)

            def __call__(self, x, *, ctx: nnx.Context):
                x = self.block1(x, ctx=ctx)
                x = self.block2(x, ctx=ctx)
                return x

        @nnx.jit_filter
        def train_step(model: Model, x, y):
            @nnx.grad
            def loss_fn(model: Model):
                ctx = nnx.Context(flags=dict(use_running_average=False))
                y_pred = model(x, ctx=ctx)
                return jnp.mean((y - y_pred) ** 2)

            grads = loss_fn(model)
            model.update[:] = jax.tree_map(
                lambda w, g: w - 0.1 * g, model.get("params"), grads
            )

            return model

        ctx = nnx.Context(jax.random.PRNGKey(0))
        model = Model(ctx=ctx)

        x = np.random.uniform(size=(4, 2))
        y = np.random.uniform(size=(4, 2))

        for _i in range(3):
            model = train_step(model, x, y)

        assert model.block1.linear.bias is not None
        assert model.block2.linear.bias is not None
        assert model.block1.linear.kernel is model.block2.linear.kernel
        assert model.block1.linear.bias is model.block2.linear.bias
        assert model.block1.bn is not model.block2.bn
