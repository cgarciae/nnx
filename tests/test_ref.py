import jax
import pytest
import typing as tp

import nnx

A = tp.TypeVar("A")


class TestRef:
    def test_slots(self):
        ref = nnx.Ref(1)
        assert not hasattr(ref, "__dict__")
        value = nnx.Value(1, None)
        assert not hasattr(value, "__dict__")
        index = nnx.Index(None)
        assert not hasattr(index, "__dict__")

    def test_ref(self):
        r1 = nnx.Ref(1)
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

    def test_value_and_index_are_deref(self):
        value = nnx.Value(1, None)
        index = nnx.Index(None)

        assert isinstance(value, nnx.Deref)
        assert isinstance(index, nnx.Deref)

    def test_ref_trace_level(self):
        r1: nnx.Ref[int] = nnx.Ref(1)

        @jax.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f()

        @jax.jit
        def g(pytree, dagdef):
            r2, r3 = nnx.reref(pytree, dagdef)
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return nnx.deref(r2)

        r2 = nnx.reref(*g(*nnx.deref((r1, r1))))

        assert r1.value == 1
        assert r2.value == 2

        r2.value = 3
        assert r1.value == 1
        assert r2.value == 3

        r3 = nnx.reref(*g(*nnx.deref((r1, r1))))

        assert r3 is not r2
        assert r3.value == 2

    def test_ref_trace_level_grad(self):
        r1: nnx.Ref[int] = nnx.Ref(1)

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
        r1 = nnx.Ref(1)
        r2 = nnx.Ref(2)

        pytree = pytree0 = {"a": [r1, r2], "b": r1}

        @jax.jit
        def f(pytree, dagdef):
            pytree = nnx.reref(pytree, dagdef)

            assert pytree["a"][0] is pytree["b"]
            assert pytree["a"][1] is not pytree["b"]

            return nnx.deref(pytree)

        pytree, dagdef = f(*nnx.deref(pytree))
        pytree = nnx.reref(pytree, dagdef)

        assert pytree["a"][0] is pytree["b"]
        assert pytree["a"][1] is not pytree["b"]

        # compare with pytree0
        assert pytree["a"][0] is not pytree0["a"][0]
        assert pytree["a"][1] is not pytree0["a"][1]
        assert pytree["b"] is not pytree0["b"]

    def test_barrier_edge_case(self):
        r1: tp.Optional[nnx.Ref[tp.Any]] = None

        @jax.jit
        def f():
            nonlocal r1
            x = jax.numpy.empty(1)
            r1 = nnx.Ref(x)
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
        r1: nnx.Ref[int] = nnx.Ref(1)

        @jax.jit
        def g(r2, dagdef):
            r2 = nnx.reref(r2, dagdef)
            r2.value += 1
            assert r1 is not r2
            return nnx.deref(r2)

        r2 = nnx.reref(*g(*nnx.deref(r1)))
        assert r1 is not r2
        assert r1.value == 1
        assert r2.value == 2

        r3 = nnx.reref(*g(*nnx.deref(r2)))
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
        r1 = nnx.Ref(1)
        r2 = nnx.Ref(2)

        @jax.jit
        def g(pytree, dagdef):
            r3, r4, r5 = nnx.reref(pytree, dagdef)
            nonlocal n
            n += 1
            assert r3 is r4
            assert r4 is not r5
            return nnx.deref(r3)

        r6 = nnx.reref(*g(*nnx.deref((r1, r1, r2))))
        assert r6 is not r1
        assert r6.value == r1.value
        assert n == 1

        g(*nnx.deref((r1, r1, r2)))
        assert n == 1

        g(*nnx.deref((r2, r2, r1)))
        assert n == 1

        with pytest.raises(AssertionError):
            g(*nnx.deref((r1, r2, r1)))

        assert n == 2

    def test_deref_number_of_fields(self):
        r1 = nnx.Ref(1)
        r2 = nnx.Ref(2)
        v1 = 3
        pytree = {
            "a": [r1, r2, v1],
            "b": {"c": r1, "d": r2},
        }
        assert len(jax.tree_util.tree_leaves(pytree)) == 5

        pytree, dagdef = nnx.deref(pytree)
        assert len(jax.tree_util.tree_leaves(pytree)) == 3

        pytree = nnx.reref(pytree, dagdef)
        assert len(jax.tree_util.tree_leaves(pytree)) == 5

    def test_mutable(self):
        r1 = nnx.Ref(1, collection="params")
        r2 = nnx.Ref(2, collection="batch_stats")

        with nnx.mutable(lambda c: c == "params"):
            r1.value = 3
            with pytest.raises(
                ValueError, match="Collection 'batch_stats' is not mutable"
            ):
                r2.value = 4

    def test_dag(self):
        r1 = nnx.Ref(1)
        r2 = nnx.Ref(2)
        v1 = 3
        pytree = {
            "a": [r1, r2, v1],
            "b": {"c": r1, "d": r2},
        }
        dag = nnx.Dag(pytree)

        @jax.jit
        def f(dag: nnx.Dag):
            dag.value["a"][0].value = 4
            return dag

        dag = f(dag)

        assert dag.value["a"][0].value == 4
        assert dag.value["b"]["c"].value == 4

    def test_clone(self):
        r1 = nnx.Ref(1)
        r2 = nnx.Ref(2)
        v1 = 3
        pytree = {
            "a": [r1, r2, v1],
            "b": {"c": r1, "d": r2},
        }

        pytree2 = nnx.clone(pytree)

        pytree["a"][0].value = 10
        assert pytree2["a"][0].value == 1
