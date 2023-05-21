import jax
import jax.numpy as jnp
import numpy as np

import nnx


class TestIntegration:
    def test_shared_modules(self):
        class Block(nnx.Module):
            def __init__(self, linear: nnx.Linear, *, rngs: nnx.Rngs):
                self.linear = linear
                self.bn = nnx.BatchNorm(2, rngs=rngs)

            def __call__(self, x, *, train: bool):
                x = self.linear(x)
                x = self.bn(x, use_running_average=not train)
                return nnx.relu(x)

        class Model(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                shared = nnx.Linear(2, 2, rngs=rngs)
                self.block1 = Block(shared, rngs=rngs)
                self.block2 = Block(shared, rngs=rngs)

            def __call__(self, x, *, train: bool):
                x = self.block1(x, train=train)
                x = self.block2(x, train=train)
                return x

        @nnx.jit
        def train_step(model: Model, x, y):
            @nnx.grad
            def loss_fn(model: Model):
                y_pred = model(x, train=True)
                return jnp.mean((y - y_pred) ** 2)

            grads = loss_fn(model)
            model[:] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grads)

        rngs = nnx.Rngs(jax.random.PRNGKey(0))
        model = Model(rngs=rngs)

        x = np.random.uniform(size=(4, 2))
        y = np.random.uniform(size=(4, 2))

        for _i in range(3):
            train_step(model, x, y)

        assert model.block1.linear.bias is not None
        assert model.block2.linear.bias is not None
        assert model.block1.linear.kernel is model.block2.linear.kernel
        assert model.block1.linear.bias is model.block2.linear.bias
        assert model.block1.bn is not model.block2.bn

    def test_shared_modules_jit_filter(self):
        class Block(nnx.Module):
            def __init__(self, linear: nnx.Linear, *, rngs: nnx.Rngs):
                self.linear = linear
                self.bn = nnx.BatchNorm(2, rngs=rngs)

            def __call__(self, x, *, train: bool):
                x = self.linear(x)
                x = self.bn(x, use_running_average=not train)
                return nnx.relu(x)

        class Model(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                shared = nnx.Linear(2, 2, rngs=rngs)
                self.block1 = Block(shared, rngs=rngs)
                self.block2 = Block(shared, rngs=rngs)

            def __call__(self, x, *, train: bool):
                x = self.block1(x, train=train)
                x = self.block2(x, train=train)
                return x

        @nnx.jit_filter
        def train_step(model: Model, x, y):
            @nnx.grad
            def loss_fn(model: Model):
                y_pred = model(x, train=True)
                return jnp.mean((y - y_pred) ** 2)

            grads = loss_fn(model)
            model[:] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grads)

            return model

        rngs = nnx.Rngs(jax.random.PRNGKey(0))
        model = Model(rngs=rngs)

        x = np.random.uniform(size=(4, 2))
        y = np.random.uniform(size=(4, 2))

        for _i in range(3):
            model = train_step(model, x, y)

        assert model.block1.linear.bias is not None
        assert model.block2.linear.bias is not None
        assert model.block1.linear.kernel is model.block2.linear.kernel
        assert model.block1.linear.bias is model.block2.linear.bias
        assert model.block1.bn is not model.block2.bn
