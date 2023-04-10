from inspect import istraceback

import jax
import numpy as np
import pytest

import nnx


class TestScope:
    def test_make_rng(self):
        with nnx.scope({"a": jax.random.PRNGKey(0)}, flags={}):
            key1 = nnx.make_rng("a")
            assert isinstance(key1, jax.Array)
            key2 = nnx.make_rng("a")
            assert isinstance(key2, jax.Array)

            assert not np.allclose(key1, key2)

    def test_make_rng_error(self):
        with pytest.raises(ValueError, match="Unknown collection:"):
            nnx.make_rng("a")

    def test_flags(self):
        with nnx.scope(flags={"b": 1}):
            b = nnx.get_flag("b")
            assert b == 1

    def test_flags_error(self):
        with pytest.raises(ValueError, match="Unknown flag:"):
            nnx.get_flag("a")
