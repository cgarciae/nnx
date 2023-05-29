from functools import partial
import typing as tp

import jax
import pytest

import nnx


def collection(collection: str):
    return lambda x: isinstance(x, nnx.Referential) and x.collection == collection


class TestFilters:
    # def test_dagify_jit(self):
    #     r1 = nnx.Ref(10.0)
    #     r2 = nnx.Ref(20.0)

    #     pytree: tp.Dict[str, tp.Any] = {
    #         "a": [r1, r2],
    #         "b": r1,
    #         "c": 7,
    #         "d": 5.0,
    #     }

    #     @nnx.dagify(jax.jit)
    #     def f(pytree):
    #         pytree["a"][0].value *= -1
    #         return pytree

    #     pytree = f(pytree)

    #     assert pytree["a"][0].value == -10.0
    #     assert pytree["a"][1].value == 20.0
    #     assert pytree["b"].value == -10.0
    #     assert pytree["c"] == 7
    #     assert pytree["d"] == 5.0

    def test_trace_level(self):
        r1: nnx.Ref[int] = nnx.param(1)

        @jax.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f()

    def test_jit(self):
        r1: nnx.Ref[int] = nnx.param(1)

        @nnx.jit_filter
        def g(m: nnx.Seq[nnx.Ref[int]]):
            r2, r3 = m
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return nnx.Seq([r2])

        r2 = g(nnx.Seq((r1, r1)))[0]

        assert r1.value == 1
        assert r2.value == 2

        r2.value = 3
        assert r1.value == 1
        assert r2.value == 3

        r3 = g(nnx.Seq((r1, r1)))[0]

        assert r3 is not r2
        assert r3.value == 2
