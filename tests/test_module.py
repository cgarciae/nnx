import nnx
import jax.numpy as jnp
import jax


class TestModule:
    def test_apply(self):
        class Foo(nnx.Module):
            w: jax.Array = nnx.param()

            def __init__(self, c: float):
                key = nnx.make_rng("params")
                self.w = jax.random.uniform(key, ())
                self.c = c

            def __call__(self, x):
                e = nnx.make_rng("e")
                return self.w * x + jax.random.normal(e, ()) + self.c

        foo = Foo.init(rngs={"params": jax.random.PRNGKey(0)})(c=1.0)

        y = foo.apply(rngs={"e": jax.random.PRNGKey(1)})(x=2.0)

        assert isinstance(y, jax.Array)


class TestModuleDef:
    def test_apply(self):
        class Foo(nnx.Module):
            w: jax.Array = nnx.param()

            def __init__(self, c: float):
                key = nnx.make_rng("params")
                self.w = jax.random.uniform(key, ())
                self.c = c

            def __call__(self, x):
                e = nnx.make_rng("e")
                return self.w * x + jax.random.normal(e, ()) + self.c

        foo = Foo.init(rngs={"params": jax.random.PRNGKey(0)})(c=1.0)

        partitions, moduledef = foo.partition()

        assert "params" in partitions
        assert "rest" in partitions

        y = moduledef.apply(partitions, rngs={"e": jax.random.PRNGKey(1)})(x=2.0)

        assert isinstance(y, jax.Array)
