import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

import nnx

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

    def init_fn(key, shape, dtype, in_axis, out_axis) -> jax.Array:
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


class Attention(nnx.Module):
    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding

        key = ctx.make_rng("params")
        self.WQ = nnx.param(
            dense_init(
                key, (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1, 2)
            ),
            P(sharding.embed, sharding.heads, sharding.depth),
        )
        key = ctx.make_rng("params")
        self.WK = nnx.param(
            dense_init(
                key, (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1, 2)
            ),
            P(sharding.embed, sharding.heads, sharding.depth),
        )
        key = ctx.make_rng("params")
        self.WV = nnx.param(
            dense_init(
                key, (cfg.embed, cfg.heads, cfg.depth), cfg.param_dtype, 0, (1, 2)
            ),
            P(sharding.embed, sharding.heads, sharding.depth),
        )
        key = ctx.make_rng("params")
        self.WO = nnx.param(
            dense_init(
                key, (cfg.heads, cfg.depth, cfg.embed), cfg.param_dtype, (0, 1), 2
            ),
            P(sharding.heads, sharding.depth, sharding.embed),
        )
        # cache
        self.index = nnx.var("cache", jnp.array(0, dtype=jnp.int32), P())
        self.key = nnx.var(
            "cache",
            jnp.zeros(
                (cfg.batch, cfg.heads, cfg.depth, cfg.max_length),
                jnp.bfloat16,
            ),
            P(sharding.batch, sharding.heads, sharding.depth, None),
        )
        self = nnx.var(
            "cache",
            jnp.zeros(
                (cfg.batch, cfg.heads, cfg.depth, cfg.max_length),
                jnp.bfloat16,
            ),
            P(sharding.batch, sharding.heads, sharding.depth, None),
        )

    # We combine the cache and params into "vs", but it would be no harder at all
    # to thread through a separate "cache" argument storing cache entries.
    def __call__(self, cfg: Config, x_q, x_kv, mask=None, *, ctx: nnx.Context):
        q = jnp.einsum("bse,enh->bsnh", x_q, self.WQ.astype(cfg.dtype)).astype(
            jnp.float32
        )
        k = jnp.einsum("bte,enh->btnh", x_kv, self.WK.astype(cfg.dtype)).astype(
            jnp.float32
        )
        v = jnp.einsum("bte,enh->btnh", x_kv, self.WV.astype(cfg.dtype))

        index = None
        if cfg.decode:
            index = self.index
            one_hot_indices = jax.nn.one_hot(
                self.index, cfg.max_length, dtype=cfg.dtype
            )
            self.key = self.key + jnp.moveaxis(k, -3, -1) * one_hot_indices
            self = self + jnp.moveaxis(v, -3, -1) * one_hot_indices
            k = jnp.moveaxis(self.key, -1, -3)
            v = jnp.moveaxis(self, -1, -3)
            cache_mask = jnp.broadcast_to(
                jnp.arange(cfg.max_length) <= self.index,
                (cfg.batch, 1, 1, cfg.max_length),
            )
            mask = jnp.logical_and(
                cache_mask if mask is None else mask, cache_mask
            ).astype(cfg.dtype)
            self.index = self.index + 1

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
        s = dropout(cfg, s, ctx=ctx)
        a = jnp.einsum("bnst,btnh->bsnh", s, v)
        o = jnp.einsum("bsnh,nhe->bse", a, self.WO.astype(cfg.dtype))

        return o


class MLP(nnx.Module):
    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding
        self.Win1 = nnx.param(
            dense_init(
                ctx.make_rng("params"), (cfg.embed, cfg.hidden), cfg.param_dtype, 0, 1
            ),
            P(sharding.embed, sharding.hidden),
        )
        self.Win2 = nnx.param(
            dense_init(
                ctx.make_rng("params"), (cfg.embed, cfg.hidden), cfg.param_dtype, 0, 1
            ),
            P(sharding.embed, sharding.hidden),
        )
        self.Wout = nnx.param(
            dense_init(
                ctx.make_rng("params"), (cfg.hidden, cfg.embed), cfg.param_dtype, 0, 1
            ),
            P(sharding.hidden, sharding.embed),
        )

    def __call__(self, cfg: Config, x, *, ctx: nnx.Context):
        h1 = jnp.einsum("bse,eh->bsh", x, self.Win1.astype(cfg.dtype))
        h2 = jnp.einsum("bse,eh->bsh", x, self.Win2.astype(cfg.dtype))
        h = jax.nn.gelu(h1) * h2
        h = dropout(cfg, h, ctx=ctx)
        o = jnp.einsum("bsh,he->bse", h, self.Wout.astype(cfg.dtype))
        return o


class DecoderBlock(nnx.Module):
    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding
        self.attn = Attention(cfg, ctx=ctx)
        self.mlp = MLP(cfg, ctx=ctx)
        self.scale1 = nnx.param(
            jnp.ones((cfg.embed,), cfg.param_dtype), P(sharding.embed)
        )
        self.scale2 = nnx.param(
            jnp.ones((cfg.embed,), cfg.param_dtype), P(sharding.embed)
        )

    def __call__(self, cfg: Config, input, *, ctx: nnx.Context):
        x = rms_norm(cfg, self.scale1, input)
        x = self.attn(cfg, x, x, mask=None, ctx=ctx)
        x = dropout(cfg, x, ctx=ctx)
        x = x + input
        y = rms_norm(cfg, self.scale2, x)
        y = self.mlp(cfg, y, ctx=ctx)
        y = dropout(cfg, y, ctx=ctx)
        return y + x


class Decoder(nnx.Module):
    def __init__(self, cfg: Config, *, ctx: nnx.Context):
        sharding = cfg.sharding
        self.embed = nnx.param(
            embed_init(
                ctx.make_rng("params"), (cfg.vocab, cfg.embed), cfg.param_dtype, 1, 0
            ),
            P(sharding.vocab, sharding.embed),
        )
        self.unembed = nnx.param(
            dense_init(
                ctx.make_rng("params"), (cfg.embed, cfg.vocab), jnp.float32, 0, 1
            ),
            P(sharding.embed, sharding.vocab),
        )
        self.scale1 = nnx.param(
            jnp.ones((cfg.embed,), cfg.param_dtype), P(sharding.embed)
        )

        if cfg.scanned:
            self.layers = jax.vmap(
                lambda key: DecoderBlock(cfg, ctx=nnx.Context(key)).partition()
            )(jax.random.split(ctx.make_rng("params"), cfg.layers)).merge()
        else:
            self.layers = nnx.Sequence(
                DecoderBlock(cfg, ctx=ctx) for _ in range(cfg.layers)
            )

    def __call__(self, cfg: Config, x, *, ctx: nnx.Context):
        # TODO: handle right-shifting for training: here or in train loop.
        # TODO: handle general mask routing.
        x = self.embed.astype(cfg.dtype)[x]

        if cfg.scanned:
            assert isinstance(self.layers, DecoderBlock)

            state, moduledef = self.layers.partition()

            def scan_fn(x, s: tp.Tuple[tp.Any, nnx.State]):
                key, state = s
                _ctx = nnx.Context(dropout=key)
                y, (state, _) = moduledef.apply(state)(cfg, x, ctx=_ctx)
                return y, state

            dropout_key = ctx.make_rng("dropout")
            x, state = jax.lax.scan(
                scan_fn,
                x,
                (jax.random.split(dropout_key, cfg.layers), state),
            )
            self.layers.update_state(state)
        else:
            assert isinstance(self.layers, nnx.Sequence)
            for decoder_block in self.layers:
                x = decoder_block(cfg, x, ctx=ctx)

        x = jnp.einsum("bse,ev->bsv", x, self.unembed)
        return x
