import nnx
import jax.numpy as jnp
import jax


class TestModule:
    def test_call(self):
        class Foo(nnx.Module):
            w: jax.Array = nnx.param()

            def __init__(self, c: float, *, ctx: nnx.Context):
                key = ctx.make_rng("params")
                self.w = jax.random.uniform(key, ())
                self.c = c

            def __call__(self, x, *, ctx: nnx.Context):
                key = ctx.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        ctx = nnx.Context(jax.random.PRNGKey(0))
        foo = Foo(c=1.0, ctx=ctx)

        ctx = nnx.Context(dict(e=jax.random.PRNGKey(1)))
        y = foo(x=2.0, ctx=ctx)

        assert isinstance(y, jax.Array)


class TestModuleDef:
    def test_apply(self):
        class Foo(nnx.Module):
            w: jax.Array = nnx.param()
            c: float = nnx.node_field()

            def __init__(self, c: float, *, ctx: nnx.Context):
                self.w = jax.random.uniform(ctx.make_rng("params"), ())
                self.c = c

            def __call__(self, x, *, ctx: nnx.Context):
                key = ctx.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        ctx = nnx.Context(jax.random.PRNGKey(0))
        foo = Foo(c=1.0, ctx=ctx)

        partitions, moduledef = foo.partition()

        assert "params" in partitions
        assert "rest" in partitions

        ctx = nnx.Context(dict(e=jax.random.PRNGKey(1)))
        y, partitions = moduledef.apply(partitions)(x=2.0, ctx=ctx)

        assert isinstance(y, jax.Array)
