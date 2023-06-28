from typing import Any

import jax
import numpy as np
import pytest

import nnx
from nnx.contextlib import _stable_hash


class TestContext:
    def test_hash(self):
        _hash = _stable_hash("hi")
        assert isinstance(_hash, int)

    def test_rng_stream(self):
        key0 = jax.random.PRNGKey(0)
        ctx = nnx.context(key0)
        assert ctx._rngs["params"].count == 0

        key1 = ctx.make_rng("params")
        assert ctx._rngs["params"].count == 1
        assert ctx._rngs["params"].key is key0
        assert not np.equal(key0, key1).all()

        key2 = ctx.make_rng("params")
        assert ctx._rngs["params"].count == 2
        assert ctx._rngs["params"].key is key0
        assert not np.equal(key1, key2).all()

    def test_rng_fork(self):
        key0 = jax.random.PRNGKey(0)
        ctx1 = nnx.context(key0)
        ctx2 = ctx1.partition().merge()

        assert ctx2._rngs["params"].count == 0
        assert ctx2._rngs["params"].count_path == (0,)

        key1 = ctx1.make_rng("params")
        key2 = ctx2.make_rng("params")

        assert not np.equal(key1, key2).all()

    def test_rng_trace_level_constraints(self):
        ctx = nnx.context(0)

        @jax.jit
        def f():
            with pytest.raises(
                nnx.TraceContextError,
                match="Cannot use Context from a different trace level",
            ):
                ctx.make_rng("params")

        f()

        @jax.jit
        def f():
            with pytest.raises(
                nnx.TraceContextError,
                match="Cannot use Context from a different trace level",
            ):
                ctx.partition()

        f()

        ctx1: Any = None

        @jax.jit
        def g():
            nonlocal ctx1
            ctx1 = nnx.context(1)

        g()

        assert isinstance(ctx1, nnx.Context)
        with pytest.raises(
            nnx.TraceContextError,
            match="Cannot use Context from a different trace level",
        ):
            ctx1.make_rng("params")

    def test_partition_merge(self):
        ctx = nnx.context(dropout=0)

        keys, ctxdef = ctx.partition()

        assert "dropout" in keys
        assert ctxdef._rng_counts == (("dropout", (0,)),)

        ctx2 = ctxdef.merge(keys)

        key1 = ctx.make_rng("dropout")
        key2 = ctx2.make_rng("dropout")
        assert not np.equal(key1, key2).all()

        ctx3 = ctxdef.merge(keys)
        key3 = ctx3.make_rng("dropout")
        assert np.equal(key2, key3).all()
