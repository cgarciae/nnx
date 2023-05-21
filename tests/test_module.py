import nnx
import jax.numpy as jnp
import jax


class TestModule:
    def test_call(self):
        class Foo(nnx.Module):
            w: jax.Array = nnx.param()

            def __init__(self, c: float, *, rngs: nnx.Rngs):
                key = rngs.make_rng("params")
                self.w = jax.random.uniform(key, ())
                self.c = c

            def __call__(self, x, *, rngs: nnx.Rngs):
                key = rngs.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        rngs = nnx.Rngs(jax.random.PRNGKey(0))
        foo = Foo(c=1.0, rngs=rngs)

        rngs = nnx.Rngs(e=jax.random.PRNGKey(1))
        y = foo(x=2.0, rngs=rngs)

        assert isinstance(y, jax.Array)


class TestModuleDef:
    def test_apply(self):
        class Foo(nnx.Module):
            w: jax.Array = nnx.param()

            def __init__(self, c: float, *, rngs: nnx.Rngs):
                self.w = jax.random.uniform(rngs.make_rng("params"), ())
                self.c = c

            def __call__(self, x, *, rngs: nnx.Rngs):
                key = rngs.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        rngs = nnx.Rngs(jax.random.PRNGKey(0))
        foo = Foo(c=1.0, rngs=rngs)

        partitions, moduledef = foo.partition()

        assert "params" in partitions
        assert "rest" in partitions

        rngs = nnx.Rngs(e=jax.random.PRNGKey(1))
        y, partitions = moduledef.apply(partitions)(x=2.0, rngs=rngs)

        assert isinstance(y, jax.Array)
