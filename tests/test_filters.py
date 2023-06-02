from functools import partial
import typing as tp

import jax
import pytest

import nnx


def collection(collection: str):
    return lambda x: isinstance(x, nnx.MutableVariable) and x.collection == collection


class TestFilters:
    def test_trace_level(self):
        r1: nnx.MutableVariable[int] = nnx.param(1)

        @jax.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f()

    def test_jit(self):
        m = nnx.Map(
            a=nnx.param(1),
            b=nnx.param(2),
        )

        @nnx.jit_filter_pure
        def g(m: nnx.Map[int]):
            m.a = 10
            return m

        m = g(m).merge()

        assert m.a == 10
        assert m.b == 2
