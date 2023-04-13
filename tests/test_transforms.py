import typing as tp
from functools import partial
import jax

import pytest
import refx

import nnx


def collection(collection: str):
    return lambda x: isinstance(x, refx.Referential) and x.collection == collection


class TestJIT:
    def test_jit(self):
        r1: refx.Ref[int] = refx.Ref(1)
        pytree = (r1, r1)

        @nnx.jit
        def f(pytree):
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f(pytree)

        @nnx.jit
        def g(pytree):
            r2, r3 = pytree
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return 1.0

        out = g(pytree)

        assert pytree[0].value == 2
        assert pytree[1].value == 2
        assert out == 1.0

    def test_jit_stateless(self):
        r1: refx.Ref[int] = refx.Ref(1)
        pytree = (r1, r1)

        @partial(nnx.jit, stateful=False)
        def g(pytree):
            r2, r3 = pytree
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return 1.0

        out = g(pytree)

        assert pytree[0].value == 1
        assert pytree[1].value == 1
        assert out == 1.0


class TestGrad:
    def test_grad(self):
        p1 = refx.Ref(10.0, "params")
        p2 = refx.Ref(20.0, "params")

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        @nnx.grad
        def f(pytree):
            # sum all params
            return pytree["a"][0].value + pytree["a"][1].value + pytree["b"].value

        grad = f(pytree)
        assert isinstance(grad, dict)

        assert grad[("a", "0")].value == 2.0
        assert isinstance(grad[("a", "0")], refx.Value)
        assert grad[("a", "1")].value == 1.0
        assert isinstance(grad[("a", "1")], refx.Value)
        assert isinstance(grad[("b",)], refx.Index)
        assert grad[("c",)] is refx.NOTHING
        assert grad[("d",)] is refx.NOTHING

        refx.update_refs(pytree, grad)
        assert pytree["a"][0].value == 2.0
        assert pytree["a"][1].value == 1.0
        assert pytree["b"].value == 2.0
        assert pytree["c"] == 7
        assert pytree["d"] == 5.0

    def test_grad_with_multiple_ref_types(self):
        p1 = refx.Ref(10.0, "params")
        p2 = refx.Ref(20.0, "batch_stats")

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        @nnx.grad
        def f(pytree):
            # sum all params
            return pytree["a"][0].value + pytree["a"][1].value + pytree["b"].value

        grad = f(pytree)
        assert isinstance(grad, dict)

        assert grad[("a", "0")].value == 2.0
        assert isinstance(grad[("a", "0")], refx.Value)
        assert grad[("a", "0")].collection == "params"
        assert grad[("a", "1")] is refx.NOTHING
        assert isinstance(grad[("b",)], refx.Index)
        assert grad[("b",)].collection == "params"
        assert grad[("c",)] is refx.NOTHING
        assert grad[("d",)] is refx.NOTHING

        refx.update_refs(refx.get_partition(pytree, collection("params")), grad)
        assert pytree["a"][0].value == 2.0
        assert pytree["a"][1].value == 20.0
        assert pytree["b"].value == 2.0
        assert pytree["c"] == 7
        assert pytree["d"] == 5.0

    def test_grad_with_type_predicate(self):
        p1 = refx.Ref(10.0, "params")
        p2 = refx.Ref(20.0, "batch_stats")

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        @partial(nnx.grad, wrt="batch_stats")
        def f(pytree):
            # sum all params
            return pytree["a"][0].value + pytree["a"][1].value + pytree["b"].value

        grad = f(pytree)
        assert isinstance(grad, dict)

        assert grad[("a", "0")] is refx.NOTHING
        assert grad[("a", "1")].value == 1.0
        assert isinstance(grad[("a", "1")], refx.Value)
        assert grad[("a", "1")].collection == "batch_stats"
        assert grad[("b",)] is refx.NOTHING
        assert grad[("c",)] is refx.NOTHING
        assert grad[("d",)] is refx.NOTHING

        refx.update_refs(refx.get_partition(pytree, collection("batch_stats")), grad)
        assert pytree["a"][0].value == 10.0
        assert pytree["a"][1].value == 1.0
        assert pytree["b"].value == 10.0
        assert pytree["c"] == 7
        assert pytree["d"] == 5.0

    def test_scope(self):
        p1 = refx.Ref(10.0, "params")
        p2 = refx.Ref(20.0, "params")

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        @nnx.grad
        def f(pytree):
            # sum all params
            return pytree["a"][0].value + pytree["a"][1].value + pytree["b"].value

        with nnx.scope({"a": jax.random.PRNGKey(0)}, flags={}):
            grad = f(pytree)
        assert isinstance(grad, dict)
