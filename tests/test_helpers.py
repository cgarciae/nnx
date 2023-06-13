import jax
import optax

import nnx


class TestHelpers:
    def test_train_state(self):
        m = nnx.Map(a=nnx.param(1), b=nnx.var("batch_stats", 2))

        (params, batch_stats), moduledef = m.partition("params", "batch_stats")

        state = nnx.TrainState(
            apply_fn=moduledef.apply,
            params=params,
            tx=optax.sgd(1.0),
            batch_stats=batch_stats,
            other=nnx.node(100),
            int=200,
            static=nnx.static(300),
        )

        leaves = jax.tree_util.tree_leaves(state)

        assert 1 in leaves
        assert 2 in leaves
        assert 100 in leaves
        assert 200 not in leaves
        assert 300 not in leaves
