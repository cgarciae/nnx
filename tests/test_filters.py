from functools import partial
import typing as tp

import jax
import pytest

import refx


def collection(collection: str):
    return lambda x: isinstance(x, refx.Referential) and x.collection == collection


class TestFilters:
    def test_dagify_jit(self):
        r1 = refx.Ref(10.0)
        r2 = refx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [r1, r2],
            "b": r1,
            "c": 7,
            "d": 5.0,
        }

        @refx.dagify(jax.jit)
        def f(pytree):
            pytree["a"][0].value *= -1
            return pytree

        pytree = f(pytree)

        assert pytree["a"][0].value == -10.0
        assert pytree["a"][1].value == 20.0
        assert pytree["b"].value == -10.0
        assert pytree["c"] == 7
        assert pytree["d"] == 5.0

    def test_dagify_jit_propagate_state(self):
        r1 = refx.Ref(10.0)
        r2 = refx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [r1, r2],
            "b": r1,
            "c": 7,
            "d": 5.0,
        }

        @refx.dagify(jax.jit, propagate_state=True)
        def f(pytree):
            pytree["a"][0].value *= -1

        f(pytree)

        assert pytree["a"][0].value == -10.0
        assert pytree["a"][1].value == 20.0
        assert pytree["b"].value == -10.0
        assert pytree["c"] == 7
        assert pytree["d"] == 5.0

    def test_jit(self):
        r1: refx.Ref[int] = refx.Ref(1)

        @refx.filter_jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f()

        @refx.filter_jit
        def g(r2: refx.Ref[int], r3: refx.Ref[int]):
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return r2

        r2 = g(r1, r1)

        assert r1.value == 1
        assert r2.value == 2

        r2.value = 3
        assert r1.value == 1
        assert r2.value == 3

        r3 = g(r1, r1)

        assert r3 is not r2
        assert r3.value == 2


class TestGrad:
    def test_grad(self):
        p1 = refx.Ref(10.0)
        p2 = refx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        @refx.filter_grad
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

        refx.update_from(pytree, grad)
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

        @partial(refx.filter_grad, predicate=collection("params"))
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

        refx.update_from(refx.get_partition(collection("params"), pytree), grad)
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

        @partial(refx.filter_grad, predicate=collection("batch_stats"))
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

        refx.update_from(refx.get_partition(collection("batch_stats"), pytree), grad)
        assert pytree["a"][0].value == 10.0
        assert pytree["a"][1].value == 1.0
        assert pytree["b"].value == 10.0
        assert pytree["c"] == 7
        assert pytree["d"] == 5.0

    def test_scope(self):
        p1 = refx.Ref(10.0)
        p2 = refx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        @refx.filter_grad
        def f(pytree):
            # sum all params
            return pytree["a"][0].value + pytree["a"][1].value + pytree["b"].value

        with refx.scope({"a": jax.random.PRNGKey(0)}):
            grad = f(pytree)
        assert isinstance(grad, dict)
