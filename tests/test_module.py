from typing import Any
import pytest
import nnx
import jax.numpy as jnp
import jax


class TestModule:
    def test_call(self):
        class Foo(nnx.Module):
            def __init__(self, c: float, *, ctx: nnx.Context):
                key = ctx.make_rng("params")
                self.w = nnx.param(jax.random.uniform(key, ()))
                self.c = jnp.asarray(c)

            def __call__(self, x, *, ctx: nnx.Context):
                key = ctx.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        ctx = nnx.Context(jax.random.PRNGKey(0))
        foo = Foo(c=1.0, ctx=ctx)

        ctx = nnx.Context(dict(e=jax.random.PRNGKey(1)))
        y = foo(x=2.0, ctx=ctx)

        assert isinstance(y, jax.Array)

    def test_shared_module(self):
        m1 = nnx.Map(a=nnx.param(1), b=nnx.param(2))
        m2 = nnx.Map(x=m1, y=m1, z=nnx.param(3))

        m3 = m2.split().merge()

        assert m3["x"] is m3["y"]
        assert m3["x"]["a"] is m3["y"]["a"]
        assert m3["x"]["b"] is m3["y"]["b"]

    def test_module_graph(self):
        class Foo(nnx.Module):
            def __init__(self):
                self.a = nnx.param(1)
                self.sub = self

        m = Foo()

        state, moddef = m.split()
        assert len(state) == 1

        m2 = moddef.merge(state)
        assert m2 is m2.sub


class TestModuleDef:
    def test_apply(self):
        class Foo(nnx.Module):
            def __init__(self, c: float, *, ctx: nnx.Context):
                self.w = nnx.param(jax.random.uniform(ctx.make_rng("params"), ()))
                self.c = jnp.asarray(c)

            def __call__(self, x, *, ctx: nnx.Context):
                key = ctx.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        ctx = nnx.Context(jax.random.PRNGKey(0))
        foo = Foo(c=1.0, ctx=ctx)

        states, moddef = foo.split()
        collections = states.get_collections()

        assert "params" in collections
        assert None in collections

        ctx = nnx.Context(dict(e=jax.random.PRNGKey(1)))
        y, updates = moddef.apply(states)(x=2.0, ctx=ctx)

        assert isinstance(y, jax.Array)

    def test_derefed_mod_apply(self):
        class Foo(nnx.Module):
            def __init__(self, c: float, *, ctx: nnx.Context):
                self.w = nnx.param(
                    jax.random.uniform(ctx.make_rng("params"), ()),
                )
                self.c = jnp.asarray(c)

            def __call__(self, x, *, ctx: nnx.Context):
                key = ctx.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        ctx = nnx.Context(jax.random.PRNGKey(0))
        foo = Foo(c=1.0, ctx=ctx)

        pure_module = foo.split()
        collections = pure_module.state.get_collections()

        assert "params" in collections
        assert None in collections

        ctx = nnx.Context(dict(e=jax.random.PRNGKey(1)))
        y, states = pure_module.apply(x=2.0, ctx=ctx)

        assert isinstance(y, jax.Array)
