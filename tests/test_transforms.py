import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import pytest

import nnx


def collection(collection: str):
    return lambda x: isinstance(x, nnx.Variable) and x.collection == collection


class TestJIT:
    def test_jit(self):
        m = nnx.Dict(a=nnx.param(1))

        @nnx.jit
        def g(m: nnx.Dict):
            m.a = 2
            return 1.0

        out = g(m)

        assert m.a == 2
        assert out == 1.0

    def test_jit_stateless(self):
        m = nnx.Dict(a=nnx.param(1))

        @partial(nnx.jit, stateful=False)
        def g(m: nnx.Dict):
            m.a = 2
            return 1.0

        out = g(m)

        assert m.a == 1
        assert out == 1.0


class TestGrad:
    def test_grad(self):
        p1 = nnx.param(10.0)
        p2 = nnx.param(20.0)

        m = nnx.Dict(
            a=nnx.Sequence([p1, p2]),
            b=p1,
            c=7,
            d=5.0,
        )

        @nnx.grad
        def f(m: nnx.Dict):
            # sum all params
            return m["a"][0] + m["a"][1] + m["b"]

        grads = f(m)

        assert isinstance(grads, nnx.State)
        assert grads[("a", "0")].value == 1.0
        assert isinstance(grads[("a", "0")], nnx.Variable)
        assert grads[("a", "1")].value == 1.0
        assert isinstance(grads[("a", "1")], nnx.Variable)
        assert grads[("b",)].value == 1.0
        assert isinstance(grads[("b",)], nnx.Variable)
        assert len(grads) == 3

        m.update_state(grads)

        assert m["a"][0] == 1.0
        assert m["a"][1] == 1.0
        assert m["b"] == 1.0
        assert m["c"] == 7
        assert m["d"] == 5.0

    def test_grad_with_multiple_ref_types(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(10.0), nnx.var("batch_stats", 20.0)]),
            b=nnx.param(10.0),
            c=7,
            d=5.0,
        )

        @nnx.grad
        def f(m: nnx.Dict):
            # sum all params
            return m.a[0] + m.a[1] + m.b

        grads = f(m)

        assert isinstance(grads, nnx.State)
        assert grads[("a", "0")].value == 1.0
        assert isinstance(grads[("a", "0")], nnx.Variable)
        assert grads[("a", "0")].collection == "params"
        assert len(grads) == 2

        m.update_state(grads)

        assert m.a[0] == 1.0
        assert m.a[1] == 20.0
        assert m.b == 1.0
        assert m.c == 7
        assert m.d == 5.0

    def test_grad_with_type_predicate(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(10.0), nnx.var("batch_stats", 20.0)]),
            b=nnx.param(10.0),
            c=7,
            d=5.0,
        )

        @partial(nnx.grad, wrt="batch_stats")
        def f(m: nnx.Dict):
            # sum all params
            return m.a[0] + m.a[1] + m.b

        grads = f(m)

        assert isinstance(grads, nnx.State)
        assert grads[("a", "1")].value == 1.0
        assert isinstance(grads[("a", "1")], nnx.Variable)
        assert grads[("a", "1")].collection == "batch_stats"
        assert len(grads) == 1

        m.update_state(grads)

        assert m.a[0] == 10.0
        assert m.a[1] == 1.0
        assert m.b == 10.0
        assert m.c == 7
        assert m.d == 5.0


class TestScan:
    def test_basic(self):
        class Block(nnx.Module):
            def __init__(self, *, ctx: nnx.Context):
                self.linear = nnx.Linear(3, 3, ctx=ctx)
                self.node = jnp.ones((2,))

            def __call__(self, x: jax.Array, _) -> tp.Tuple[jax.Array, None]:
                jax.debug.print("x={x}", x=x)
                x = self.linear(x)
                x = nnx.gelu(x)
                return x, None

        MLP = nnx.scan(
            Block, variable_axes={"params": 0}, split_rngs="params", length=5
        )

        module = MLP(ctx=nnx.context(0))

        assert module.module.linear.kernel.shape == (5, 3, 3)
        assert module.module.linear.bias.shape == (5, 3)
        assert module.module.node.shape == (2,)

        x = jnp.ones((1, 3))
        y, out = module.call(x, None)

        assert y.shape == (1, 3)
        assert out is None
