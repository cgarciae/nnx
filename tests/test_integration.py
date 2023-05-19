import jax
import jax.numpy as jnp
import numpy as np

import nnx


class TestIntegration:
    def test_shared_modules(self):
        class Block(nnx.Module):
            def __init__(self, linear: nnx.Linear):
                self.linear = linear
                self.bn = nnx.BatchNorm(2)

            def __call__(self, x):
                return nnx.relu(self.bn(self.linear(x)))

        class Model(nnx.Module):
            def __init__(self):
                shared = nnx.Linear(2, 2)
                self.block1 = Block(shared)
                self.block2 = Block(shared)

            def __call__(self, x):
                x = self.block1(x)
                x = self.block2(x)
                return x

        @nnx.jit
        def train_step(model: Model, x, y):
            @nnx.grad
            def loss_fn(model: Model):
                y_pred = model.apply(use_running_average=False)(x)
                return jnp.mean((y - y_pred) ** 2)

            grad = loss_fn(model)
            model[...] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)

        model = Model.init(jax.random.PRNGKey(0))()

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
            def __init__(self, linear: nnx.Linear):
                self.linear = linear
                self.bn = nnx.BatchNorm(2)

            def __call__(self, x):
                return nnx.relu(self.bn(self.linear(x)))

        class Model(nnx.Module):
            def __init__(self):
                shared = nnx.Linear(2, 2)
                self.block1 = Block(shared)
                self.block2 = Block(shared)

            def __call__(self, x):
                x = self.block1(x)
                x = self.block2(x)
                return x

        @nnx.jit_filter
        def train_step(model: Model, x, y):
            @nnx.grad
            def loss_fn(model: Model):
                y_pred = model.apply(use_running_average=False)(x)
                return jnp.mean((y - y_pred) ** 2)

            grad = loss_fn(model)
            model[:] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)

            return model

        model = Model.init(jax.random.PRNGKey(0))()

        x = np.random.uniform(size=(4, 2))
        y = np.random.uniform(size=(4, 2))

        for _i in range(3):
            model = train_step(model, x, y)

        assert model.block1.linear.bias is not None
        assert model.block2.linear.bias is not None
        assert model.block1.linear.kernel is model.block2.linear.kernel
        assert model.block1.linear.bias is model.block2.linear.bias
        assert model.block1.bn is not model.block2.bn
