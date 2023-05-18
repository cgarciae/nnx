import jax
import numpy as np
import pytest
import nnx
from nnx.rng_stream import _stable_hash


class TestRngStream:
    def test_hash(self):
        _hash = _stable_hash("hi")
        assert isinstance(_hash, int)

    def test_rng_stream(self):
        key0 = jax.random.PRNGKey(0)
        rng = nnx.RngStream(key0)
        assert rng.count == 0

        key1 = rng.next()
        assert rng.count == 1
        assert rng.key is key0
        assert not np.equal(key0, key1).all()

        key2 = rng.next()
        assert rng.count == 2
        assert rng.key is key0
        assert not np.equal(key1, key2).all()

    def test_rng_fork(self):
        key0 = jax.random.PRNGKey(0)
        rng = nnx.RngStream(key0)

        rng1 = rng.fork()
        assert rng1.count == 0
        assert rng1.count_path == (0,)

        key1 = rng1.next()
        key2 = rng.next()

        assert not np.equal(key1, key2).all()

    def test_rng_is_pytree(self):
        key0 = jax.random.PRNGKey(0)
        rng = nnx.RngStream(key0).fork()

        rng1 = jax.tree_util.tree_map(lambda x: x, rng)

        assert rng1.count == 0
        assert rng1.count_path == (0,)
        assert rng1.key is rng.key

    def test_rng_trace_level_constraints(self):
        key0 = jax.random.PRNGKey(0)
        rng = nnx.RngStream(key0)

        @jax.jit
        def f():
            with pytest.raises(ValueError, match="Rng used in a different trace"):
                key1 = rng.key

        f()

        rng1 = None

        @jax.jit
        def g():
            nonlocal rng1
            rng1 = nnx.RngStream(jax.random.PRNGKey(1))

        g()

        with pytest.raises(ValueError, match="Rng used in a different trace"):
            key1 = rng1.key
