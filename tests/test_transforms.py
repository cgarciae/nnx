import typing as tp
from functools import partial
import jax

import pytest

import nnx


def collection(collection: str):
    return lambda x: isinstance(x, nnx.Referential) and x.collection == collection


class TestJIT:
    def test_jit(self):
        r1: nnx.Ref[int] = nnx.param(1)
        m = nnx.Seq((r1, r1))

        @nnx.jit
        def f(m):
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f(m)

        @nnx.jit
        def g(m: nnx.Seq):
            r2, r3 = m
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return 1.0

        out = g(m)

        assert m[0].value == 2
        assert m[1].value == 2
        assert out == 1.0

    def test_jit_stateless(self):
        r1 = nnx.param(1)
        m = nnx.Seq((r1, r1))

        @partial(nnx.jit, stateful=False)
        def g(m: nnx.Seq):
            r2, r3 = m
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return 1.0

        out = g(m)

        assert m[0].value == 1
        assert m[1].value == 1
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
            return m["a"][0].value + m["a"][1].value + m["b"].value

        grads = f(m)

        assert isinstance(grads, nnx.State)
        assert grads[("a", "0")].value == 2.0
        assert isinstance(grads[("a", "0")], nnx.Value)
        assert grads[("a", "1")].value == 1.0
        assert isinstance(grads[("a", "1")], nnx.Value)
        assert isinstance(grads[("b",)], nnx.Index)
        assert len(grads) == 3

        m.update(grads)

        assert m["a"][0].value == 2.0
        assert m["a"][1].value == 1.0
        assert m["b"].value == 2.0
        assert m["c"] == 7
        assert m["d"] == 5.0

    def test_grad_with_multiple_ref_types(self):
        p1 = nnx.Ref(10.0, collection="params")
        p2 = nnx.Ref(20.0, collection="batch_stats")

        m = nnx.Map(
            a=nnx.Seq([p1, p2]),
            b=p1,
            c=7,
            d=5.0,
        )

        @nnx.grad
        def f(m: nnx.Map):
            # sum all params
            return m["a"][0].value + m["a"][1].value + m["b"].value

        grads = f(m)

        assert isinstance(grads, nnx.State)
        assert grads[("a", "0")].value == 2.0
        assert isinstance(grads[("a", "0")], nnx.Value)
        assert grads[("a", "0")].collection == "params"
        assert isinstance(grads[("b",)], nnx.Index)
        assert grads[("b",)].collection == "params"
        assert len(grads) == 2

        m.update(grads)

        assert m["a"][0].value == 2.0
        assert m["a"][1].value == 20.0
        assert m["b"].value == 2.0
        assert m["c"] == 7
        assert m["d"] == 5.0

    def test_grad_with_type_predicate(self):
        p1 = nnx.param(10.0)
        p2 = nnx.ref("batch_stats", 20.0)

        m = nnx.Map(
            a=nnx.Seq([p1, p2]),
            b=p1,
            c=7,
            d=5.0,
        )

        @partial(nnx.grad, wrt="batch_stats")
        def f(m: nnx.Map):
            # sum all params
            return m["a"][0].value + m["a"][1].value + m["b"].value

        grads = f(m)

        assert isinstance(grads, nnx.State)
        assert grads[("a", "1")].value == 1.0
        assert isinstance(grads[("a", "1")], nnx.Value)
        assert grads[("a", "1")].collection == "batch_stats"
        assert len(grads) == 1

        m.update(grads)

        assert m["a"][0].value == 10.0
        assert m["a"][1].value == 1.0
        assert m["b"].value == 10.0
        assert m["c"] == 7
        assert m["d"] == 5.0

    def test_scope(self):
        p1 = nnx.param(10.0)
        p2 = nnx.param(20.0)

        m = nnx.Map(
            a=nnx.Seq([p1, p2]),
            b=p1,
            c=7,
            d=5.0,
        )
        ctx = nnx.Context(dict(a=jax.random.PRNGKey(0)))

        @nnx.grad
        def f(m: nnx.Map):
            # sum all params
            noise = jax.random.normal(ctx.make_rng("a"), shape=())
            return m["a"][0].value + m["a"][1].value + m["b"].value + noise

        grad = f(m)
        assert isinstance(grad, nnx.State)
