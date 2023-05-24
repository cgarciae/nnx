import dataclasses
import jax
import nnx
import typing as tp
import jax.numpy as jnp

ShardSpec = tp.Union[str, tp.Tuple[str,...], None]

# Sharding
@dataclasses.dataclass
class Sharding:
  batch: ShardSpec = 'data'
  sequence: ShardSpec = None
  layers : ShardSpec = None
  vocab : ShardSpec = 'model'
  embed : ShardSpec = None
  heads : ShardSpec = 'model'
  depth : ShardSpec = None
  hidden : ShardSpec = 'model'

# Config
@dataclasses.dataclass
class Config:
  # mode
  decode: bool = False
  # shapes
  batch : int = 16
  layers : int = 2
  vocab : int = 1024
  embed : int = 64
  heads : int = 12
  depth : int = 64
  hidden : int = 256
  max_length : int = 256
  # dtypes
  param_dtype : tp.Any = jnp.float32
  dtype : tp.Any = jnp.float32
  # sharding
  sharding : Sharding = Sharding()
  scanned : bool = False
  # layer params
  epsilon : float = 1e-6
  dropout_rate : float = 0.0
  rp_num_buckets : int = 32
  rp_max_distance : int = 128

cfg = Config()


def nd_dense_init(scale, mode, distribution):
    """Initializer with in_axis, out_axis set at call time."""

    def init_fn(key, shape, dtype, in_axis, out_axis):
        fn = jax.nn.initializers.variance_scaling(
            scale, mode, distribution, in_axis, out_axis
        )
        return fn(key, shape, dtype)

    return init_fn


dense_init = nd_dense_init(1.0, "fan_in", "truncated_normal")
embed_init = nd_dense_init(1.0, "fan_in", "normal")

class Attention(nnx.Module):

    WQ: jax.Array = nnx.param()
    WK: jax.Array = nnx.param()
    WV: jax.Array = nnx.param()
    WO: jax.Array = nnx.param()

    def __init__(self, cfg: Config, *, ctx: nnx.Context):
       self.WQ = Param(
          dense_init(ctx.make_rng('params'), (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1,2)), 
          P(sharding.embed, sharding.heads, sharding.depth))
      WK = Param(
          dense_init(ctx.make_rng('params'), (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1,2)), 
          P(sharding.embed, sharding.heads, sharding.depth)),
      WV = Param(
          dense_init(ctx.make_rng('params'), (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1,2)),
          P(sharding.embed, sharding.heads, sharding.depth)),
      WO = Param(
          dense_init(ctx.make_rng('params'), (cfg.heads, cfg.depth, cfg.embed), cfg.param_dtype, (0,1), 2),
          P(sharding.heads, sharding.depth, sharding.embed))