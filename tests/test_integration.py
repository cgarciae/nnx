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
            model.update_state(
                jax.tree_map(lambda w, g: w - 0.1 * g, model.filter("params"), grads)
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

    def test_shared_modules_pure(self):
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

        @jax.jit
        def train_step(pure_module: nnx.PureModule[Model], x, y):
            model = pure_module.merge()

            @nnx.grad
            def loss_fn(model: Model):
                ctx = nnx.Context(flags=dict(use_running_average=False))
                y_pred = model(x, ctx=ctx)
                return jnp.mean((y - y_pred) ** 2)

            grads = loss_fn(model)
            model.update_state(
                jax.tree_map(lambda w, g: w - 0.1 * g, model.filter("params"), grads)
            )

            return model.partition()

        ctx = nnx.Context(jax.random.PRNGKey(0))
        pure_module = Model(ctx=ctx).partition()

        x = np.random.uniform(size=(4, 2))
        y = np.random.uniform(size=(4, 2))

        for _i in range(3):
            pure_module = train_step(pure_module, x, y)

        model = pure_module.merge()

        assert model.block1.linear.bias is not None
        assert model.block2.linear.bias is not None
        assert model.block1.linear.kernel is model.block2.linear.kernel
        assert model.block1.linear.bias is model.block2.linear.bias
        assert model.block1.bn is not model.block2.bn

    def test_stateful_example(self):
        class Linear(nnx.Module):
            def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
                key = ctx.make_rng("params")
                self.w = nnx.param(jax.random.uniform(key, (din, dout)))
                self.b = nnx.param(jnp.zeros((dout,)))
                self.count = nnx.var("state", 0)

            def __call__(self, x):
                self.count += 1
                return x @ self.w + self.b

        ctx = nnx.Context(params=jax.random.PRNGKey(0))
        model = Linear(din=12, dout=2, ctx=ctx)
        # forward pass
        x = jnp.ones((8, 12))
        y = model(x)
        assert model.count == 1

        @nnx.jit
        def train_step(model, x, y):
            def loss_fn(model):
                y_pred = model(x)
                return jax.numpy.mean((y_pred - y) ** 2)

            # compute gradient
            grads: nnx.State = nnx.grad(loss_fn, wrt="params")(model)
            # SGD update
            model.update_state(
                jax.tree_map(lambda w, g: w - 0.1 * g, model.filter("params"), grads)
            )

        # execute the training step
        train_step(model, x, y)
        assert model.count == 2

    def test_functional_example(self):
        class Linear(nnx.Module):
            def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
                key = ctx.make_rng("params")
                self.w = nnx.param(jax.random.uniform(key, (din, dout)))
                self.b = nnx.param(jnp.zeros((dout,)))
                self.count = nnx.var("state", 0)

            def __call__(self, x):
                self.count += 1
                return x @ self.w + self.b

        ctx = nnx.Context(params=jax.random.PRNGKey(0))
        model = Linear(din=12, dout=2, ctx=ctx)
        # forward pass
        x = jnp.ones((8, 12))
        y = model(x)
        assert model.count == 1

        (params, state), moduledef = model.partition("params", "state")

        @jax.jit
        def train_step(params, state, x, y):
            def loss_fn(params):
                y_pred, updates = moduledef.apply(params, state)(x)
                new_state = updates.filter("state")
                loss = jax.numpy.mean((y_pred - y) ** 2)
                return loss, new_state

            # compute gradient
            grads, state = jax.grad(loss_fn, has_aux=True)(params)
            # SGD update
            params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)

            return params, state

        # execute the training step
        params, state = train_step(params, state, x, y)
        model = moduledef.merge(params, state)
        assert model.count == 2
