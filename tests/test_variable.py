import typing as tp

import jax
import pytest

import nnx

A = tp.TypeVar("A")


class TestVariable:

  def test_value(self):
    r1 = nnx.Node(1)
    assert r1.value == 1

    r2 = jax.tree_map(lambda x: x + 1, r1)

    assert r1.value == 1
    assert r2.value == 2
    assert r1 is not r2
