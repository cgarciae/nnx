import dataclasses
import jax
import nnx
import typing as tp
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

ShardSpec = tp.Union[str, tp.Tuple[str, ...], None]


# Sharding
@dataclasses.dataclass
class Sharding:
    batch: ShardSpec = "data"
    sequence: ShardSpec = None
    layers: ShardSpec = None
    vocab: ShardSpec = "model"
    embed: ShardSpec = None
    heads: ShardSpec = "model"
    depth: ShardSpec = None
    hidden: ShardSpec = "model"


# Config
@dataclasses.dataclass
class Config:
    # mode
    decode: bool = False
    # shapes
    batch: int = 16
    layers: int = 2
    vocab: int = 1024
    embed: int = 64
    heads: int = 12
    depth: int = 64
    hidden: int = 256
    max_length: int = 256
    # dtypes
    param_dtype: tp.Any = jnp.float32
    dtype: tp.Any = jnp.float32
    # sharding
    sharding: Sharding = Sharding()
    scanned: bool = False
    # layer params
    epsilon: float = 1e-6
    dropout_rate: float = 0.0
    rp_num_buckets: int = 32
    rp_max_distance: int = 128


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


def make_attention_mask(
    query_input: tp.Any,
    key_input: tp.Any,
    pairwise_fn: tp.Callable = jnp.multiply,
    dtype: tp.Any = jnp.float32,
):
    mask = pairwise_fn(
        jnp.expand_dims(query_input, axis=-1), jnp.expand_dims(key_input, axis=-2)
    )
    return jnp.expand_dims(mask, axis=-3).astype(dtype)


def make_causal_mask(x, dtype=jnp.float32):
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(idxs, idxs, jnp.greater_equal, dtype=dtype)


# padding mask
# make_attention_mask(decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype)
# packing mask
# make_attention_mask(decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype)


def sine_table(features, length, min_timescale=1.0, max_timescale=10000.0):
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here, bfloat16 rounding is catastrophic.
    sinusoid_inp = jnp.einsum(
        "i,j->ij",
        jnp.arange(length),
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


def apply_rotary_embedding(q, k, cos, sin, index=None):
    """Helper function to apply Rotary Embeddings."""
    batch, qlen, qheads, d = q.shape
    kbatch, klen, kheads, kd = k.shape
    if index is not None:
        qcos = jax.lax.broadcast_in_dim(cos[index, :], (batch, qlen, qheads, d), (3,))
        qsin = jax.lax.broadcast_in_dim(sin[index, :], (batch, qlen, qheads, d), (3,))
    else:
        qcos = jax.lax.broadcast_in_dim(cos[:qlen, :], (batch, qlen, qheads, d), (1, 3))
        qsin = jax.lax.broadcast_in_dim(sin[:qlen, :], (batch, qlen, qheads, d), (1, 3))
    kcos = jax.lax.broadcast_in_dim(cos[:klen, :], (batch, klen, kheads, d), (1, 3))
    ksin = jax.lax.broadcast_in_dim(sin[:klen, :], (batch, klen, kheads, d), (1, 3))
    out_q = (q * qcos) + (rotate_half(q) * qsin)
    out_k = (k * kcos) + (rotate_half(k) * ksin)
    return out_q, out_k


def rms_norm(cfg, scale, x):
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * jax.lax.rsqrt(mean2 + cfg.epsilon), cfg.dtype)
    return y * jnp.asarray(scale, cfg.dtype)


def dropout(cfg: Config, x, broadcast_dims=(-2,), *, ctx: nnx.Context):
    if cfg.dropout_rate == 0.0:
        return x
    broadcast_shape = list(x.shape)
    for dim in broadcast_dims:
        broadcast_shape[dim] = 1
    keep_rate = 1.0 - cfg.dropout_rate
    key = ctx.make_rng("dropout")
    mask = jax.random.bernoulli(key, p=keep_rate, shape=broadcast_shape)
    return jax.lax.select(
        jnp.broadcast_to(mask, x.shape), x / keep_rate, jnp.zeros_like(x)
    )


# def init_attn(cfg, ks):
#   sharding = cfg.sharding
#   return Tree(
#       WQ = Param(
#           dense_init(ks.WQ(), (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1,2)),
#           P(sharding.embed, sharding.heads, sharding.depth)),
#       WK = Param(
#           dense_init(ks.WK(), (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1,2)),
#           P(sharding.embed, sharding.heads, sharding.depth)),
#       WV = Param(
#           dense_init(ks.WV(), (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1,2)),
#           P(sharding.embed, sharding.heads, sharding.depth)),
#       WO = Param(
#           dense_init(ks.WO(), (cfg.heads, cfg.depth, cfg.embed), cfg.param_dtype, (0,1), 2),
#           P(sharding.heads, sharding.depth, sharding.embed))
#   )


class Attention(nnx.Module):
    WQ: jax.Array = nnx.param()
    WK: jax.Array = nnx.param()
    WV: jax.Array = nnx.param()
    WO: jax.Array = nnx.param()

    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding

        self.WQ = nnx.ref_metadata(
            dense_init(
                ctx.make_rng("params"),
                (cfg.embed, cfg.heads, cfg.depth),
                cfg.param_dtype,
                0,
                (1, 2),
            ),
            P(sharding.embed, sharding.heads, sharding.depth),
        )
        self.WK = nnx.ref_metadata(
            dense_init(
                ctx.make_rng("params"),
                (cfg.embed, cfg.heads, cfg.depth),
                cfg.param_dtype,
                0,
                (1, 2),
            ),
            P(sharding.embed, sharding.heads, sharding.depth),
        )
        self.WV = nnx.ref_metadata(
            dense_init(
                ctx.make_rng("params"),
                (cfg.embed, cfg.heads, cfg.depth),
                cfg.param_dtype,
                0,
                (1, 2),
            ),
            P(sharding.embed, sharding.heads, sharding.depth),
        )
        self.WO = nnx.ref_metadata(
            dense_init(
                ctx.make_rng("params"),
                (cfg.heads, cfg.depth, cfg.embed),
                cfg.param_dtype,
                (0, 1),
                2,
            ),
            P(sharding.heads, sharding.depth, sharding.embed),
        )

    # We combine the cache and params into "vs", but it would be no harder at all
    # to thread through a separate "cache" argument storing cache entries.
    def __call__(
        self,
        cfg: Config,
        x_q,
        x_kv,
        cache: tp.Optional["CacheLayer"] = None,
        mask=None,
        *,
        ctx: nnx.Context
    ):
        q = jnp.einsum("bse,enh->bsnh", x_q, self.WQ.astype(cfg.dtype)).astype(
            jnp.float32
        )
        k = jnp.einsum("bte,enh->btnh", x_kv, self.WK.astype(cfg.dtype)).astype(
            jnp.float32
        )
        v = jnp.einsum("bte,enh->btnh", x_kv, self.WV.astype(cfg.dtype))

        index = None
        if cfg.decode:
            assert cache is not None
            index = cache.index
            one_hot_indices = jax.nn.one_hot(
                cache.index, cfg.max_length, dtype=cfg.dtype
            )
            cache.key = cache.key + jnp.moveaxis(k, -3, -1) * one_hot_indices
            cache.value = cache.value + jnp.moveaxis(v, -3, -1) * one_hot_indices
            k = jnp.moveaxis(cache.key, -1, -3)
            v = jnp.moveaxis(cache.value, -1, -3)
            cache_mask = jnp.broadcast_to(
                jnp.arange(cfg.max_length) <= cache.index,
                (cfg.batch, 1, 1, cfg.max_length),
            )
            mask = jnp.logical_and(
                cache_mask if mask is None else mask, cache_mask
            ).astype(cfg.dtype)
            cache.index = cache.index + 1

        attention_bias = 0.0
        if mask is None:  # Hack in lieu of general mask routing.
            mask = make_causal_mask(x, jnp.float32)
        if mask is not None:
            attention_bias = jax.lax.select(
                mask > 0,
                jnp.full(mask.shape, 0.0, cfg.dtype),
                jnp.full(mask.shape, -1e10, cfg.dtype),
            )

        sin, cos = sine_table(q.shape[-1], max(q.shape[1], k.shape[1]))
        q, k = apply_rotary_embedding(q, k, cos, sin, index=index)

        l = jnp.einsum("bsnh,btnh->bnst", q, k) / np.sqrt(cfg.depth) + attention_bias
        s = jax.nn.softmax(l).astype(cfg.dtype)
        s = dropout(cfg, ks.attn_dropout, s, ctx=ctx)
        a = jnp.einsum("bnst,btnh->bsnh", s, v)
        o = jnp.einsum("bsnh,nhe->bse", a, vs.WO.astype(cfg.dtype))

        return o, vs


# def init_mlp(cfg, ks):
#   sharding = cfg.sharding
#   return Tree(
#     Win1 = Param(
#         dense_init(ks.Win1(), (cfg.embed, cfg.hidden), cfg.param_dtype, 0, 1),
#         P(sharding.embed, sharding.hidden)),
#     Win2 = Param(
#         dense_init(ks.Win2(), (cfg.embed, cfg.hidden), cfg.param_dtype, 0, 1),
#         P(sharding.embed, sharding.hidden)),
#     Wout = Param(
#         dense_init(ks.Wout(), (cfg.hidden, cfg.embed), cfg.param_dtype, 0, 1),
#         P(sharding.hidden, sharding.embed))
#   )


class MLP(nnx.Module):
    Win1: jax.Array = nnx.param()
    Win2: jax.Array = nnx.param()
    Wout: jax.Array = nnx.param()

    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding
        self.Win1 = nnx.ref_metadata(
            dense_init(
                ctx.make_rng("params"), (cfg.embed, cfg.hidden), cfg.param_dtype, 0, 1
            ),
            P(sharding.embed, sharding.hidden),
        )
        self.Win2 = nnx.ref_metadata(
            dense_init(
                ctx.make_rng("params"), (cfg.embed, cfg.hidden), cfg.param_dtype, 0, 1
            ),
            P(sharding.embed, sharding.hidden),
        )
        self.Wout = nnx.ref_metadata(
            dense_init(
                ctx.make_rng("params"), (cfg.hidden, cfg.embed), cfg.param_dtype, 0, 1
            ),
            P(sharding.hidden, sharding.embed),
        )


# def init_decoder_block(cfg, ks):
#   sharding = cfg.sharding
#   attn_vs = init_attn(cfg, ks.attn)
#   ff_vs = init_mlp(cfg, ks.mlp)
#   ln_vs = Tree(
#     scale1 = Param(jnp.ones((cfg.embed,), cfg.param_dtype), P(sharding.embed)),
#     scale2 = Param(jnp.ones((cfg.embed,), cfg.param_dtype), P(sharding.embed))
#   )
#   return attn_vs | ff_vs | ln_vs


class DecoderBlock(nnx.Module):
    ln1: jax.Array = nnx.param()
    ln2: jax.Array = nnx.param()

    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding
        self.attn = Attention(cfg, ctx=ctx)
        self.mlp = MLP(cfg, ctx=ctx)
        self.scale1 = nnx.ref_metadata(
            jnp.ones((cfg.embed,), cfg.param_dtype), P(sharding.embed)
        )
        self.scale2 = nnx.ref_metadata(
            jnp.ones((cfg.embed,), cfg.param_dtype), P(sharding.embed)
        )


# def init_decoder(cfg, ks):
#   sharding = cfg.sharding
#   vs = Tree(
#       embed=Param(
#           embed_init(ks.embed(), (cfg.vocab, cfg.embed), cfg.param_dtype, 1, 0),
#           P(sharding.vocab, sharding.embed)),
#       unembed = Param(
#           dense_init(ks.unembed(), (cfg.embed, cfg.vocab), jnp.float32, 0, 1),  # check this init
#           P(sharding.embed, sharding.vocab)),
#       scale1 = Param(
#           jnp.ones((cfg.embed,), cfg.param_dtype),
#           P(sharding.embed,))
#   )
#   if cfg.scanned:
#     _, scanned_vs = jax.lax.scan(
#         lambda _, key: (_, init_decoder_block(cfg, Tumbler(key))),
#         None,
#         random.split(ks.layers(), cfg.layers))
#     vs = vs.at.layers.set(scanned_vs)
#   else:
#     for idx in range(cfg.layers):
#       vs = vs.at.layers[idx].set(init_decoder_block(cfg, ks.layers[idx]))
#   return vs


class Decoder(nnx.Module):
    embed: jax.Array = nnx.param()
    unembed: jax.Array = nnx.param()
    scale1: jax.Array = nnx.param()
    layers: tp.Union[DecoderBlock, tp.Tuple[DecoderBlock, ...]] = nnx.node_field()

    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding
        self.embed = nnx.ref_metadata(
            embed_init(
                ctx.make_rng("params"),
                (cfg.vocab, cfg.embed),
                cfg.param_dtype,
                1,
                0,
            ),
            P(sharding.vocab, sharding.embed),
        )
        self.unembed = nnx.ref_metadata(
            dense_init(
                ctx.make_rng("params"),
                (cfg.embed, cfg.vocab),
                jnp.float32,
                0,
                1,
            ),
            P(sharding.embed, sharding.vocab),
        )
        self.scale1 = nnx.ref_metadata(
            jnp.ones((cfg.embed,), cfg.param_dtype), P(sharding.embed)
        )

        if cfg.scanned:
            self.layers = jax.vmap(
                lambda key: DecoderBlock(cfg, ctx=nnx.Context(key)).deref()
            )(jax.random.split(ctx.make_rng("params"), cfg.layers)).reref()
        else:
            self.layers = tuple(DecoderBlock(cfg, ctx=ctx) for _ in range(cfg.layers))


# def init_cache(cfg):
#   def init_cache_layer(cfg):
#     sharding = cfg.sharding
#     return Tree(
#       index = Cache(jnp.array(0, dtype=jnp.int32), P()),
#       key = Cache(
#           jnp.zeros((cfg.batch, cfg.heads, cfg.depth, cfg.max_length), jnp.bfloat16),
#           P(sharding.batch, sharding.heads, sharding.depth, None)),
#       value = Cache(
#           jnp.zeros((cfg.batch, cfg.heads, cfg.depth, cfg.max_length), jnp.bfloat16),
#           P(sharding.batch, sharding.heads, sharding.depth, None))
#   )
#   vs = Tree()
#   if cfg.scanned:
#     _, scanned_vs = jax.lax.scan(
#         lambda c, s: (c, init_cache_layer(cfg)),
#         None, None, length=cfg.layers)
#     vs = vs.at.layers.set(scanned_vs)
#   else:
#     for idx in range(cfg.layers):
#       vs = vs.at.layers[idx].set(init_cache_layer(cfg))
#   return vs


class CacheLayer(nnx.Module):
    index: jax.Array = nnx.ref("cache")
    key: jax.Array = nnx.ref("cache")
    value: jax.Array = nnx.ref("cache")

    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding
        self.index = nnx.ref_metadata(jnp.array(0, dtype=jnp.int32), P())
        self.key = nnx.ref_metadata(
            jnp.zeros(
                (cfg.batch, cfg.heads, cfg.depth, cfg.max_length),
                jnp.bfloat16,
            ),
            P(sharding.batch, sharding.heads, sharding.depth, None),
        )
        self.value = nnx.ref_metadata(
            jnp.zeros(
                (cfg.batch, cfg.heads, cfg.depth, cfg.max_length),
                jnp.bfloat16,
            ),
            P(sharding.batch, sharding.heads, sharding.depth, None),
        )


class Cache(nnx.Module):
    layers: tp.Union[CacheLayer, tp.Tuple[CacheLayer, ...]] = nnx.node_field()

    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        if cfg.scanned:
            self.layers = jax.vmap(
                lambda key: CacheLayer(cfg, ctx=nnx.Context(key)).deref()
            )(jax.random.split(ctx.make_rng("params"), cfg.layers)).reref()
        else:
            self.layers = tuple(CacheLayer(cfg, ctx=ctx) for _ in range(cfg.layers))
