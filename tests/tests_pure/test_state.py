import jax
import pure


class TestState:
    def test_pytree(self):
        state = pure.State(
            a=pure.Variable(1),
            b=pure.Variable(2),
        )

        state = jax.tree_map(lambda x: x * 2, state)

        assert state.a == 2
        assert state.b == 4

    def test_update(self):
        state = pure.State(
            a=pure.Variable(1),
            b=pure.Variable(2),
        )

        state = state.update(a=3)

        assert state.a == 3
        assert state.b == 2
