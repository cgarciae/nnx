import jax
import numpy as np
import pytest

import nnx


class TestContainers:
    def test_node_idenpotence(self):
        x = nnx.node(1)
        x = nnx.node(x)

        assert isinstance(x, nnx.Node)

    def test_variable_idenpotence(self):
        x = nnx.var("x", 1)
        x = nnx.var("x", x)

        assert isinstance(x, nnx.Variable)
        assert x.collection == "x"

    def test_variable_cannot_change_collection(self):
        x = nnx.var("x", 1)

        with pytest.raises(ValueError, match="is not equivalent to"):
            x = nnx.var("y", x)

    def test_container_cannot_change_type(self):
        x = nnx.var("x", 1)

        with pytest.raises(ValueError, match="is not equivalent to"):
            x = nnx.node(x)

        x = nnx.node(2)

        with pytest.raises(ValueError, match="is not equivalent to"):
            x = nnx.var("x", x)

    def test_static_is_empty(self):
        leaves = jax.tree_util.tree_leaves(nnx.Static(1))

        assert len(leaves) == 0

    def test_static_empty_pytree(self):
        static = nnx.Static(2)

        static = jax.tree_map(lambda x: x + 1, static)

        assert static.value == 2

    def test_static_array_not_jitable(self):
        @jax.jit
        def f(x):
            return x

        # first time you don't get an error due to a bug in jax
        f(nnx.Static(np.random.uniform(size=(10, 10))))

        with pytest.raises(ValueError):
            f(nnx.Static(np.random.uniform(size=(10, 10))))
