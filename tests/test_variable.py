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

    

    def test_deref_through_jit(self):
        r1 = nnx.Variable(1, "", None)
        r2 = nnx.Variable(2, "", None)

        m = m0 = nnx.Map({"a": nnx.Seq([r1, r2]), "b": r1})

        @jax.jit
        def f(pure_module: nnx.PureModule[nnx.Map[tp.Any]]):
            m = pure_module.merge()

            assert m["a"][0] is not m["b"]
            assert m["a"][1] is not m["b"]

            return m.split()

        m = f(m.split()).merge()

        assert m["a"][0] is not m["b"]
        assert m["a"][1] is not m["b"]

        # compare with pytree0
        assert m["a"][0] is not m0["a"][0]
        assert m["a"][1] is not m0["a"][1]
        assert m["b"] is not m0["b"]

    def test_barrier_edge_case(self):
        r1: tp.Optional[nnx.Variable[tp.Any]] = None

        @jax.jit
        def f():
            nonlocal r1
            x = jax.numpy.empty(1)
            r1 = nnx.Variable(x, "", None)
            return x

        x = f()
        assert r1 is not None

        with pytest.raises(
            ValueError, match="Cannot mutate ref from different trace level"
        ):
            r1.value = 2

        @jax.jit
        def g():
            nonlocal r1
            assert r1 is not None
            x = jax.numpy.empty(1)
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = x
            return x

        x = g()

    def test_cross_barrier(self):
        m = nnx.Map(a=nnx.param(1))

        @jax.jit
        def g(pure_module: nnx.PureModule[nnx.Map[int]]):
            m = pure_module.merge()
            m.a += 1
            return m.split()

        m2 = g(m.split()).merge()
        assert m2 is not m
        assert m.a == 1
        assert m2.a == 2

    def test_no_rejit(self):
        n = 0
        m = nnx.Map(a=nnx.param(1))

        @jax.jit
        def g(pure_module):
            nonlocal n
            n += 1
            m = pure_module.merge()
            m.a += 1
            return m.split()

        m2 = g(m.split()).merge()

        assert n == 1
        assert m2 is not m
        assert m.a == 1
        assert m2.a == 2

        g(m.split())
        assert n == 1

        g(m2.split())
        assert n == 1

        m2.b = nnx.param(10)
        g(m2.split())

        assert n == 2

    def test_deref_number_of_fields(self):
        r1 = nnx.Variable(1, "", None)
        r2 = nnx.Variable(2, "", None)
        v1 = 3
        m = nnx.Map(
            {
                "a": nnx.Seq([r1, r2, v1]),
                "b": nnx.Map({"c": r1, "d": r2}),
            }
        )

        p, moduledef = m.split()
        assert len(p) == 4
        assert len(jax.tree_util.tree_leaves(p)) == 4

    def test_deref_arrays_are_nodes(self):
        # test arrays are nodes
        r1 = nnx.Variable(1, "", None)
        r2 = nnx.Variable(2, "", None)
        v1 = jax.numpy.array(3)
        m = nnx.Map(
            {
                "a": nnx.Seq([r1, r2, v1]),
                "b": nnx.Map({"c": r1, "d": r2}),
            }
        )

        p, moduledef = m.split()
        assert len(p) == 5
        assert len(jax.tree_util.tree_leaves(p)) == 5

    @pytest.mark.skip(reason="TODO: removing support for now")
    def test_mutable(self):
        r1 = nnx.Variable(1, "params", None)
        r2 = nnx.Variable(2, "batch_stats", None)

        with nnx.mutable(lambda c: c == "params"):
            r1.value = 3
            with pytest.raises(
                ValueError, match="Collection 'batch_stats' is not mutable"
            ):
                r2.value = 4

    def test_clone(self):
        m = nnx.Map(
            a=nnx.Seq([nnx.param(1), nnx.param(2), 3]),
            b=nnx.Map(c=nnx.param(1), d=nnx.param(2)),
        )

        m2 = m.clone()

        assert m is not m2
        assert m2.a[0] == m2.b.c
        assert m2.a[1] == m2.b.d

        assert m.a[0] == m2.a[0]
        assert m.a[1] == m2.a[1]
        assert m.b.c == m2.b.c
        assert m.b.d == m2.b.d
