import typing as tp
from functools import partial
import jax

import pytest

import nnx


def collection(collection: str):
    return lambda x: isinstance(x, nnx.Variable) and x.collection == collection


class TestJIT:
    def test_jit(self):
        m = nnx.Map(a=nnx.param(1))

        @nnx.jit
        def g(m: nnx.Map):
            m.a = 2
            return 1.0

        out = g(m)

        assert m.a == 2
        assert out == 1.0

    def test_jit_stateless(self):
        m = nnx.Map(a=nnx.param(1))

        @partial(nnx.jit, stateful=False)
        def g(m: nnx.Map):
            m.a = 2
            return 1.0

        out = g(m)

        assert m.a == 1
        assert out == 1.0


class TestGrad:
    def test_grad(self):
        p1 = nnx.param(10.0)
        p2 = nnx.param(20.0)

        m = nnx.Map(
            a=nnx.Seq([p1, p2]),
            b=p1,
            c=7,
            d=5.0,
        )

        @nnx.grad
        def f(m: nnx.Map):
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
        m = nnx.Map(
            a=nnx.Seq([nnx.param(10.0), nnx.var("batch_stats", 20.0)]),
            b=nnx.param(10.0),
            c=7,
            d=5.0,
        )

        @nnx.grad
        def f(m: nnx.Map):
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
        m = nnx.Map(
            a=nnx.Seq([nnx.param(10.0), nnx.var("batch_stats", 20.0)]),
            b=nnx.param(10.0),
            c=7,
            d=5.0,
        )

        @partial(nnx.grad, wrt="batch_stats")
        def f(m: nnx.Map):
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
