import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

import nnx

A = tp.TypeVar("A")


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
        ctx = nnx.context(flags=dict(use_running_average=False))
        y_pred = model(x, ctx=ctx)
        return jnp.mean((y - y_pred) ** 2)

      grads = loss_fn(model)
      model.update_state(
          jax.tree_map(lambda w, g: w - 0.1 * g, model.filter(nnx.Param), grads)
      )

    model = Model(ctx=nnx.context(0))

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
        ctx = nnx.context(flags=dict(use_running_average=False))
        y_pred = model(x, ctx=ctx)
        return jnp.mean((y - y_pred) ** 2)

      grads = loss_fn(model)
      model.update_state(
          jax.tree_map(lambda w, g: w - 0.1 * g, model.filter(nnx.Param), grads)
      )

      return model.partition()

    pure_module = Model(ctx=nnx.context(0)).partition()

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
    class State(nnx.Variable[A]):
      pass

    class Linear(nnx.Module):

      def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        key = ctx.make_rng("params")
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.count = State(0)

      def __call__(self, x):
        self.count += 1
        return x @ self.w + self.b

    model = Linear(din=12, dout=2, ctx=nnx.context(0))
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
      grads: nnx.State = nnx.grad(loss_fn, wrt=nnx.Param)(model)
      # SGD update
      model.update_state(
          jax.tree_map(lambda w, g: w - 0.1 * g, model.filter(nnx.Param), grads)
      )

    # execute the training step
    train_step(model, x, y)
    assert model.count == 2

  def test_functional_example(self):
    class Count(nnx.Variable[A]):
      pass

    class Linear(nnx.Module):

      def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        key = ctx.make_rng("params")
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.count = Count(0)

      def __call__(self, x):
        self.count += 1
        return x @ self.w + self.b

    model = Linear(din=12, dout=2, ctx=nnx.context(0))
    # forward pass
    x = jnp.ones((8, 12))
    y = model(x)
    assert model.count == 1

    (params, counts), moduledef = model.partition(nnx.Param, Count)

    @jax.jit
    def train_step(params, counts, x, y):
      def loss_fn(params):
        y_pred, (updates, _) = moduledef.apply(params, counts)(x)
        loss = jax.numpy.mean((y_pred - y) ** 2)
        return loss, updates.filter(Count)

      # compute gradient
      grads, counts = jax.grad(loss_fn, has_aux=True)(params)
      # SGD update
      params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)

      return params, counts

    # execute the training step
    params, counts = train_step(params, counts, x, y)
    model = moduledef.merge(params, counts)
    assert model.count == 2

  def test_intermediates_example(self):
    class Linear(nnx.Module):

      def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        key = ctx.make_rng("params")
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

      def __call__(self, x):
        y = x @ self.w + self.b
        self.y = nnx.Intermediate(y)
        return y

    model = Linear(12, 2, ctx=nnx.context(0))

    y = model(jnp.ones((8, 12)))

    intermediates = model.pop_state(nnx.Intermediate)

    assert "y" in intermediates

  def test_intermediates_example_functional(self):
    class Linear(nnx.Module):

      def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        key = ctx.make_rng("params")
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

      def __call__(self, x):
        y = x @ self.w + self.b
        self.y = nnx.Intermediate(y)
        return y

    model = Linear(12, 2, ctx=nnx.context(0))

    state, moduledef = model.partition()

    y, (state, _) = moduledef.apply(state)(jnp.ones((8, 12)))

    intermediates, state = state.partition(nnx.Intermediate, ...)

    assert "y" in intermediates
