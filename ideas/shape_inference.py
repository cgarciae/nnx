import typing as tp

import jax
import jax.numpy as jnp
from jax import random

import nnx


class Linear(nnx.Module):

  @tp.overload
  def __init__(self, *, din: int, dout: int, ctx: nnx.Context):
    ...

  @tp.overload
  def __init__(self, *, dout: int):
    ...

  @tp.overload
  def __init__(
      self,
      *,
      din: tp.Optional[int] = None,
      dout: int,
      ctx: tp.Optional[nnx.Context] = None,
  ):
    ...

  def __init__(
      self,
      *,
      din: tp.Optional[int] = None,
      dout: int,
      ctx: tp.Optional[nnx.Context] = None,
  ):
    self.dout = dout
    if din is not None:
      if ctx is None:
        raise ValueError("ctx must be provided if din is provided")
      self.init_variables(din, ctx)

  def init_variables(self, din: int, ctx: nnx.Context):
    key = ctx.make_rng("params")
    self.w = nnx.Param(random.uniform(key, (din, self.dout)))
    self.b = nnx.Param(jnp.zeros((self.dout,)))

  def __call__(
      self, x: jax.Array, *, ctx: tp.Optional[nnx.Context] = None
  ) -> jax.Array:
    if self.is_initializing and not hasattr(self, "w"):
      if ctx is None:
        raise ValueError("ctx must be provided to initialize module")
      self.init_variables(x.shape[-1], ctx)

    return x @ self.w + self.b


class BatchNorm(nnx.Module):

  @tp.overload
  def __init__(self, *, mu: float = 0.95):
    ...

  @tp.overload
  def __init__(self, *, din: int, mu: float = 0.95, ctx: nnx.Context):
    ...

  @tp.overload
  def __init__(
      self,
      *,
      din: tp.Optional[int] = None,
      mu: float = 0.95,
      ctx: tp.Optional[nnx.Context] = None,
  ):
    ...

  def __init__(
      self,
      *,
      din: tp.Optional[int] = None,
      mu: float = 0.95,
      ctx: tp.Optional[nnx.Context] = None,
  ):
    self.mu = mu

    if din is not None:
      if ctx is None:
        raise ValueError("ctx must be provided if din is provided")
      self.init_variables(din, ctx)

  def init_variables(self, din: int, ctx: nnx.Context):
    self.scale = nnx.Param(jax.numpy.ones((din,)))
    self.bias = nnx.Param(jax.numpy.zeros((din,)))
    self.mean = nnx.BatchStat(jax.numpy.zeros((din,)))
    self.var = nnx.BatchStat(jax.numpy.ones((din,)))

  def __call__(
      self, x, *, train: bool, ctx: tp.Optional[nnx.Context] = None
  ) -> jax.Array:
    if self.is_initializing and not hasattr(self, "scale"):
      if ctx is None:
        raise ValueError("ctx must be provided to initialize module")
      self.init_variables(x.shape[-1], ctx)

    if train:
      axis = tuple(range(x.ndim - 1))
      mean = jax.numpy.mean(x, axis=axis)
      var = jax.numpy.var(x, axis=axis)
      # ema update
      self.mean = self.mu * self.mean + (1 - self.mu) * mean
      self.var = self.mu * self.var + (1 - self.mu) * var
    else:
      mean, var = self.mean, self.var

    scale, bias = self.scale, self.bias
    x = (x - mean) / jax.numpy.sqrt(var + 1e-5) * scale + bias
    return x


class Dropout(nnx.Module):

  def __init__(self, rate: float):
    self.rate = rate

  def __call__(self, x: jax.Array, *, train: bool, ctx: nnx.Context) -> jax.Array:
    if train:
      mask = random.bernoulli(ctx.make_rng("dropout"), (1 - self.rate), x.shape)
      x = x * mask / (1 - self.rate)
    return x


# ----------------------------
# test Linear
# ----------------------------
print("test Linear")

# eager
m1 = Linear(din=32, dout=10, ctx=nnx.context(params=0))
y = m1(x=jnp.ones((1, 32)))
print(jax.tree_map(jnp.shape, m1.get_state()))

# lazy
m2 = Linear(dout=10)
y = m2.init(x=jnp.ones((1, 32)), ctx=nnx.context(params=0))
print(jax.tree_map(jnp.shape, m2.get_state()))

# usage
y1 = m1(x=jnp.ones((1, 32)))
y2 = m2(x=jnp.ones((1, 32)))

# ----------------------------
# Test scan
# ----------------------------
print("\ntest scan")


class Block(nnx.Module):

  def __init__(
      self,
      din: tp.Optional[int] = None,
      dout: int = 10,
      ctx: tp.Optional[nnx.Context] = None,
  ):
    self.linear = Linear(din=din, dout=dout, ctx=ctx)
    self.bn = BatchNorm(din=dout if din is not None else None, ctx=ctx)
    self.dropout = Dropout(0.5)

  def __call__(self, x: jax.Array, _, *, train: bool, ctx: nnx.Context):
    x = self.linear(x, ctx=ctx)
    x = self.bn(x, train=train, ctx=ctx)
    x = self.dropout(x, train=train, ctx=ctx)
    x = jax.nn.gelu(x)
    return x, None


MLP = nnx.Scan(
    Block,
    variable_axes={nnx.Param: 0},
    variable_carry=nnx.BatchStat,
    split_rngs={"params": True, "dropout": True},
    length=5,
)


# eager
mlp = MLP(din=10, dout=10, ctx=nnx.context(params=0))
y, _ = mlp.call(jnp.ones((1, 10)), None, train=True, ctx=nnx.context(dropout=1))
print(f"{y.shape=}")
print("state =", jax.tree_map(jnp.shape, mlp.get_state()))
print()

# lazy
mlp = MLP(dout=10)
mlp.init(jnp.ones((1, 10)), None, train=False, ctx=nnx.context(params=0))
y, _ = mlp.call(jnp.ones((1, 10)), None, train=True, ctx=nnx.context(dropout=1))
print(f"{y.shape=}")
print("state =", jax.tree_map(jnp.shape, mlp.get_state()))
