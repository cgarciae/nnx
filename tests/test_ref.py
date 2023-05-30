import jax
import pytest
import typing as tp

import nnx

A = tp.TypeVar("A")


class TestRef:
    def test_slots(self):
        ref = nnx.Ref(1, "")
        assert not hasattr(ref, "__dict__")
        value = nnx.Value(1, "", None)
        assert not hasattr(value, "__dict__")

    def test_ref(self):
        r1 = nnx.Ref(1, "")
        assert r1.value == 1

        def add_one(r):
            r.value += 1
            return r

        r2 = jax.tree_map(add_one, r1)

        assert r1.value == 2
        assert r2.value == 2
        assert r1 is r2

        r1.value = 3

        assert r1.value == 3
        assert r2.value == 3

    def test_ref_trace_level(self):
        r1: nnx.Ref[int] = nnx.Ref(1, "")

        @jax.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f()

        @jax.jit
        def g(derefed: nnx.DerefedMod[nnx.Partition, nnx.Seq[tp.Any]]):
            r2, r3 = derefed.reref()

            r2.value = 2
            assert r1 is not r2
            return nnx.Seq([r2]).deref()

        m = nnx.Seq((r1, r1))
        r2 = g(m.deref()).reref()[0]

        assert r1.value == 1
        assert r2.value == 2

        r2.value = 3
        assert r1.value == 1
        assert r2.value == 3

        r3 = g(nnx.Seq((r1, r1)).deref()).reref()[0]

        assert r3 is not r2
        assert r3.value == 2

    def test_ref_trace_level_grad(self):
        r1: nnx.Ref[int] = nnx.Ref(1, "")

        @jax.grad
        def f(w):
            with pytest.raises(
                ValueError,
                match="Cannot mutate ref with value that contains tracers from other",
            ):
                r1.value = w
            return 1.0

        f(3.0)

    def test_deref_through_jit(self):
        r1 = nnx.Ref(1, "")
        r2 = nnx.Ref(2, "")

        m = m0 = nnx.Map({"a": nnx.Seq([r1, r2]), "b": r1})

        @jax.jit
        def f(dermod: nnx.DerefedMod[nnx.Partition, nnx.Map[tp.Any]]):
            m = dermod.reref()

            assert m["a"][0] is not m["b"]
            assert m["a"][1] is not m["b"]

            return m.deref()

        m = f(m.deref()).reref()

        assert m["a"][0] is not m["b"]
        assert m["a"][1] is not m["b"]

        # compare with pytree0
        assert m["a"][0] is not m0["a"][0]
        assert m["a"][1] is not m0["a"][1]
        assert m["b"] is not m0["b"]

    def test_barrier_edge_case(self):
        r1: tp.Optional[nnx.Ref[tp.Any]] = None

        @jax.jit
        def f():
            nonlocal r1
            x = jax.numpy.empty(1)
            r1 = nnx.Ref(x, "")
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
        r1: nnx.Ref[int] = nnx.Ref(1, "")

        @jax.jit
        def g(dermod: nnx.DerefedMod[nnx.Partition, nnx.Seq[tp.Any]]):
            r2 = dermod.reref()[0]
            r2.value += 1
            assert r1 is not r2
            return nnx.Seq([r2]).deref()

        r2 = g(nnx.Seq([r1]).deref()).reref()[0]
        assert r1 is not r2
        assert r1.value == 1
        assert r2.value == 2

        r3 = g(nnx.Seq([r2]).deref()).reref()[0]
        assert r1 is not r2
        assert r2 is not r3
        assert r1.value == 1
        assert r2.value == 2
        assert r3.value == 3

        # test passing a reference to a jitted function without cross_barrier
        @jax.jit
        def f(r1):
            return None

        with pytest.raises(TypeError, match="Cannot interpret value of type"):
            f(r1)

        assert isinstance(r1.value, int)
        assert r1.value == 1

    def test_no_rejit(self):
        n = 0
        r1 = nnx.Ref(1, "a")
        r2 = nnx.Ref(2, "b")

        @jax.jit
        def g(dermod):
            r3, r4, r5 = dermod.reref()
            nonlocal n
            n += 1
            assert r3 is not r4
            assert r4 is not r5
            return nnx.Seq([r3]).deref()

        r6 = g(nnx.Seq((r1, r1, r2)).deref()).reref()[0]
        assert r6 is not r1
        assert r6.value == r1.value
        assert n == 1

        g(nnx.Seq((r1, r1, r2)).deref())
        assert n == 1

        g(nnx.Seq((r2, r2, r1)).deref())
        assert n == 2

        g(nnx.Seq((r1, r1, r2)).deref())

        assert n == 2

    def test_deref_number_of_fields(self):
        r1 = nnx.Ref(1, "")
        r2 = nnx.Ref(2, "")
        v1 = 3
        m = nnx.Map(
            {
                "a": nnx.Seq([r1, r2, v1]),
                "b": nnx.Map({"c": r1, "d": r2}),
            }
        )

        p, moddef = m.deref()
        assert len(p) == 4
        assert len(jax.tree_util.tree_leaves(p)) == 4

    def test_deref_arrays_are_nodes(self):
        # test arrays are nodes
        r1 = nnx.Ref(1, "")
        r2 = nnx.Ref(2, "")
        v1 = jax.numpy.array(3)
        m = nnx.Map(
            {
                "a": nnx.Seq([r1, r2, v1]),
                "b": nnx.Map({"c": r1, "d": r2}),
            }
        )

        p, moddef = m.deref()
        assert len(p) == 5
        assert len(jax.tree_util.tree_leaves(p)) == 5

    @pytest.mark.skip(reason="TODO: removing support for now")
    def test_mutable(self):
        r1 = nnx.Ref(1, collection="params")
        r2 = nnx.Ref(2, collection="batch_stats")

        with nnx.mutable(lambda c: c == "params"):
            r1.value = 3
            with pytest.raises(
                ValueError, match="Collection 'batch_stats' is not mutable"
            ):
                r2.value = 4

    def test_clone(self):
        r1 = nnx.Ref(1, "")
        r2 = nnx.Ref(2, "")
        v1 = 3
        m = nnx.Map(
            {
                "a": nnx.Seq([r1, r2, v1]),
                "b": nnx.Map({"c": r1, "d": r2}),
            }
        )

        m2 = m.clone()

        assert m2["a"][0] is not m2["b"]["c"]
        assert m2["a"][1] is not m2["b"]["d"]

        assert m["a"][0] is not m2["a"][0]
        assert m["a"][1] is not m2["a"][1]
        assert m["b"]["c"] is not m2["b"]["c"]
        assert m["b"]["d"] is not m2["b"]["d"]
