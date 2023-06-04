import jax
import pytest
import typing as tp

import nnx

A = tp.TypeVar("A")


class TestVariable:
    def test_slots(self):
        ref = nnx.Variable(1, "", None)
        assert not hasattr(ref, "__dict__")

    def test_value(self):
        r1 = nnx.Variable(1, "", None)
        assert r1.value == 1

        r2 = jax.tree_map(lambda x: x + 1, r1)

        assert r1.value == 1
        assert r2.value == 2
        assert r1 is not r2
