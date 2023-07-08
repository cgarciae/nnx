from typing import Any

import jax
import jax.numpy as jnp
import pytest

import nnx


class TestModule:
    def test_has_module_state(self):
        class Foo(nnx.Module):
            ...

        foo = Foo()

        assert hasattr(foo, "_module__state")

    def test_trace_level(self):
        m = nnx.Dict(a=nnx.Param(1))

        @jax.jit
        def f():
            with pytest.raises(
                nnx.TraceContextError,
                match="Cannot mutate Module from different trace level",
            ):
                m.a = 2

        f()

    def test_split_merge(self):
        m = nnx.Dict(a=nnx.Param(1))

        @jax.jit
        def g(pure_module: nnx.PureModule[nnx.Dict[int]]):
            m = pure_module.merge()
            m.a = 2
            return m.partition()

        m2 = g(m.partition()).merge()

        assert m2.a == 2

    def test_no_trace_level_error_on_grad(self):
        # No trace level error occurs because jax doesn't update
        # its top trace for grad.
        m = nnx.Dict(a=nnx.Param(1.0))

        @jax.grad
        def f(_):
            m.a = 2.0
            return 1.0

        f(1.0)

    def test_trace_level_error_on_nnx_grad(self):
        # error occurs because nnx updates its nnx_trace
        # in nnx.grad.
        m = nnx.Dict(a=nnx.Param(1.0))

        @nnx.grad
        def f(_):
            with pytest.raises(
                nnx.TraceContextError,
                match="Cannot mutate Module from different trace level",
            ):
                m.a = 2.0
            return 1.0

        f(m)

    def test_call(self):
        class Foo(nnx.Module):
            def __init__(self, c: float, *, ctx: nnx.Context):
                key = ctx.make_rng("params")
                self.w = nnx.Param(jax.random.uniform(key, ()))
                self.c = jnp.asarray(c)

            def __call__(self, x, *, ctx: nnx.Context):
                key = ctx.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        foo = Foo(c=1.0, ctx=nnx.context(0))

        y = foo(x=2.0, ctx=nnx.context(e=1))

        assert isinstance(y, jax.Array)

    def test_shared_module(self):
        m1 = nnx.Dict(a=nnx.Param(1), b=nnx.Param(2))
        m2 = nnx.Dict(x=m1, y=m1, z=nnx.Param(3))

        m3 = m2.partition().merge()

        assert m3["x"] is m3["y"]
        assert m3["x"]["a"] is m3["y"]["a"]
        assert m3["x"]["b"] is m3["y"]["b"]

    def test_module_graph(self):
        class Foo(nnx.Module):
            def __init__(self):
                self.a = nnx.Param(1)
                self.sub = self

        m = Foo()

        state, moduledef = m.partition()
        assert len(state) == 1

        m2 = moduledef.merge(state)
        assert m2 is m2.sub

    def test_deref_through_jit(self):
        r1 = nnx.Node(1)
        r2 = nnx.Node(2)

        m = m0 = nnx.Dict({"a": nnx.Sequence([r1, r2]), "b": r1})

        @jax.jit
        def f(pure_module: nnx.PureModule[nnx.Dict[Any]]):
            m = pure_module.merge()

            assert m["a"][0] is not m["b"]
            assert m["a"][1] is not m["b"]

            return m.partition()

        m = f(m.partition()).merge()

        assert m["a"][0] is not m["b"]
        assert m["a"][1] is not m["b"]

        # compare with pytree0
        assert m["a"][0] is not m0["a"][0]
        assert m["a"][1] is not m0["a"][1]
        assert m["b"] is not m0["b"]

    def test_cross_barrier(self):
        m = nnx.Dict(a=nnx.Param(1))

        @jax.jit
        def g(pure_module: nnx.PureModule[nnx.Dict[int]]):
            m = pure_module.merge()
            m.a += 1
            return m.partition()

        m2 = g(m.partition()).merge()
        assert m2 is not m
        assert m.a == 1
        assert m2.a == 2

    def test_no_rejit(self):
        n = 0
        m = nnx.Dict(a=nnx.Param(1))

        @jax.jit
        def g(pure_module):
            nonlocal n
            n += 1
            m = pure_module.merge()
            m.a += 1
            return m.partition()

        m2 = g(m.partition()).merge()

        assert n == 1
        assert m2 is not m
        assert m.a == 1
        assert m2.a == 2

        g(m.partition())
        assert n == 1

        g(m2.partition())
        assert n == 1

        m2.b = nnx.Param(10)
        g(m2.partition())

        assert n == 2

    def test_deref_number_of_fields(self):
        r1 = nnx.Node(1)
        r2 = nnx.Node(2)
        v1 = 3
        m = nnx.Dict(
            {
                "a": nnx.Sequence([r1, r2, v1]),
                "b": nnx.Dict({"c": r1, "d": r2}),
            }
        )

        p, moduledef = m.partition()
        assert len(p) == 4
        assert len(jax.tree_util.tree_leaves(p)) == 4

    def test_deref_arrays_are_nodes(self):
        # test arrays are nodes
        r1 = nnx.Node(1)
        r2 = nnx.Node(2)
        v1 = jax.numpy.array(3)
        m = nnx.Dict(
            {
                "a": nnx.Sequence([r1, r2, v1]),
                "b": nnx.Dict({"c": r1, "d": r2}),
            }
        )

        p, moduledef = m.partition()
        assert len(p) == 5
        assert len(jax.tree_util.tree_leaves(p)) == 5

    def test_clone(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), 3]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.Param(2)),
        )

        m2 = m.clone()

        assert m is not m2
        assert m2.a[0] == m2.b.c
        assert m2.a[1] == m2.b.d

        assert m.a[0] == m2.a[0]
        assert m.a[1] == m2.a[1]
        assert m.b.c == m2.b.c
        assert m.b.d == m2.b.d

    def test_sow_basic(self):
        class Foo(nnx.Module):
            def __call__(self, x):
                y = x + 1
                self.sow(nnx.Intermediate, "y", y)
                return y

        m = Foo()
        y1 = m(2)
        y2 = m(10)

        assert y1 == 3
        assert y2 == 11
        assert m.y == (3, 11)

        intermediates = m.pop_state(nnx.Intermediate)

        assert isinstance(intermediates["y"], nnx.Intermediate)
        assert intermediates["y"].value == (3, 11)

        assert not hasattr(m, "y")

    def test_sow_existing_non_variable_field(self):
        class Foo(nnx.Module):
            def __init__(self) -> None:
                self.y = 10

            def __call__(self, x):
                y = x + 1
                self.sow(nnx.Intermediate, "y", y)
                return y

        m = Foo()

        with pytest.raises(ValueError, match="to be a Variable, got"):
            m(2)

    def test_sow_wrong_collection(self):
        class Foo(nnx.Module):
            def __init__(self) -> None:
                self.y = nnx.Param(10)

            def __call__(self, x):
                y = x + 1
                self.sow(nnx.Intermediate, "y", y)
                return y

        m = Foo()

        with pytest.raises(ValueError, match="to be of type"):
            m(2)

    def test_sow_non_tuple(self):
        class Foo(nnx.Module):
            def __init__(self) -> None:
                self.y = nnx.Intermediate(10)

            def __call__(self, x):
                y = x + 1
                self.sow(nnx.Intermediate, "y", y)
                return y

        m = Foo()

        with pytest.raises(ValueError, match="to be a tuple,"):
            m(2)


class TestModuleDataclass:
    def test_basic(self):
        @nnx.dataclass
        class Foo(nnx.Module):
            a: int = nnx.static_field()
            b: int = nnx.node_field()
            c: int = nnx.param_field()
            d: int = nnx.var_field(nnx.BatchStat)
            e: int
            f: int

        m = Foo(
            a=1,  # static
            b=2,  # node
            c=3,  # param
            d=4,  # var
            e=5,  # static int
            f=nnx.Node(6),  # test that we can pass in a node
        )

        state, moduledef = m.partition()

        assert len(state) == 4
        assert state["b"] == nnx.Node(2)
        assert state["c"] == nnx.Param(3)
        assert state["d"] == nnx.BatchStat(4)
        assert state["f"] == nnx.Node(6)

    def test_no_override(self):
        @nnx.dataclass
        class Foo(nnx.Module):
            a: int = nnx.node_field()

        with pytest.raises(ValueError, match="is not compatible with return type"):
            _m = Foo(a=nnx.Param(1))

        _m = Foo(a=nnx.Node(1))


class TestModuleDef:
    def test_apply(self):
        class Foo(nnx.Module):
            def __init__(self, c: float, *, ctx: nnx.Context):
                self.w = nnx.Param(jax.random.uniform(ctx.make_rng("params"), ()))
                self.c = jnp.asarray(c)

            def __call__(self, x, *, ctx: nnx.Context):
                key = ctx.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        ctx = nnx.context(0)
        foo = Foo(c=1.0, ctx=ctx)

        states, moduledef = foo.partition()

        assert isinstance(states, nnx.State)
        assert isinstance(states["w"], nnx.Param)
        assert isinstance(states["c"], jax.Array)

        y, _updates = moduledef.apply(states)(x=2.0, ctx=nnx.context(e=1))

        assert isinstance(y, jax.Array)

    def test_derefed_mod_apply(self):
        class Foo(nnx.Module):
            def __init__(self, c: float, *, ctx: nnx.Context):
                self.w = nnx.Param(
                    jax.random.uniform(ctx.make_rng("params"), ()),
                )
                self.c = jnp.asarray(c)

            def __call__(self, x, *, ctx: nnx.Context):
                key = ctx.make_rng("e")
                return self.w * x + jax.random.normal(key, ()) + self.c

        foo = Foo(c=1.0, ctx=nnx.context(0))

        pure_module = foo.partition()

        assert isinstance(pure_module, nnx.Pure)
        assert isinstance(pure_module.states, nnx.State)
        assert isinstance(pure_module.states["w"], nnx.Param)
        assert isinstance(pure_module.states["c"], jax.Array)

        y, states = pure_module.apply(x=2.0, ctx=nnx.context(e=1))

        assert isinstance(y, jax.Array)


class TestPureModule:
    def test_partition_merge(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Node(3)]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
        )

        pure_module = state, moduledef = m.partition()

        m2 = pure_module.merge()

        assert isinstance(state, nnx.State)
        assert isinstance(moduledef, nnx.ModuleDef)
        assert isinstance(m2, nnx.Dict)
        assert isinstance(m2.a, nnx.Sequence)
        assert isinstance(m2.b, nnx.Dict)
        assert len(m.get_state()) == 5
        assert len(m2.get_state()) == 5

    def test_partition_merge_with_filters(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Node(3)]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
        )

        pure_module = (params, batch_stats, rest), moduledef = m.partition(
            nnx.Param, nnx.BatchStat, ...
        )

        m2 = pure_module.merge()

        assert isinstance(params, nnx.State)
        assert isinstance(batch_stats, nnx.State)
        assert isinstance(rest, nnx.State)
        assert isinstance(moduledef, nnx.ModuleDef)
        assert isinstance(m2, nnx.Dict)
        assert isinstance(m2.a, nnx.Sequence)
        assert isinstance(m2.b, nnx.Dict)
        assert len(m.get_state()) == 5
        assert len(m2.get_state()) == 5

    def test_filter(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Node(3)]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
        )

        pure_module = m.partition()

        params = pure_module.filter(nnx.Param)
        batch_stats = pure_module.filter(nnx.BatchStat)
        rest = pure_module.filter(nnx.Not(nnx.Variable))

        assert len(params) == 3
        assert len(batch_stats) == 1
        assert len(rest) == 1

    def test_filter_with_filters(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Node(3)]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
        )

        pure_module = m.partition(nnx.Param, ...)

        params = pure_module.filter(nnx.Param)
        batch_stats = pure_module.filter(nnx.BatchStat)
        rest = pure_module.filter(nnx.Not(nnx.Variable))

        assert len(params) == 3
        assert len(batch_stats) == 1
        assert len(rest) == 1

    def test_partition_partition(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Node(3)]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
        )

        pure_module = m.partition()

        assert isinstance(pure_module, nnx.Pure)
        assert isinstance(pure_module.states, nnx.State)

        pure_module = pure_module.partition()

        assert isinstance(pure_module, nnx.Pure)
        assert isinstance(pure_module.states, nnx.State)

    def test_partition_with_filters_partition(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Node(3)]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
        )

        pure_module = m.partition(nnx.Param, ...)

        assert isinstance(pure_module, nnx.Pure)
        assert isinstance(pure_module.states, tuple)

        pure_module = pure_module.partition()

        assert isinstance(pure_module, nnx.Pure)
        assert isinstance(pure_module.states, nnx.State)

    def test_partition_with_filters_partition_with_filters(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Node(3)]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
        )

        pure_module = m.partition(nnx.Param, ...)

        assert isinstance(pure_module, nnx.Pure)
        assert isinstance(pure_module.states, tuple)

        pure_module = pure_module.partition(nnx.BatchStat, ...)

        assert isinstance(pure_module, nnx.Pure)
        assert isinstance(pure_module.states, tuple)

    def test_pop(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Node(3)]),
            b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
        )

        pure_module = m.partition()

        params, pure_module2 = pure_module.pop_state(nnx.Param)

        assert isinstance(params, nnx.State)
        assert isinstance(pure_module2, nnx.Pure)
        assert isinstance(pure_module2.states, nnx.State)
        assert len(params) == 3
        assert len(pure_module2.states) == 2

        (params, batch_stats), pure_module2 = pure_module.pop_state(
            nnx.Param, nnx.BatchStat
        )

        assert isinstance(params, nnx.State)
        assert isinstance(batch_stats, nnx.State)
        assert isinstance(pure_module2, nnx.Pure)
        assert isinstance(pure_module2.states, nnx.State)
        assert len(params) == 3
        assert len(batch_stats) == 1
        assert len(pure_module2.states) == 1

    def test_on_all(self):
        class Bar(nnx.Module):
            def __init__(self):
                self.a = nnx.Param(1)

        class Foo(nnx.Module):
            def __init__(self, bar):
                self.bar1 = bar
                self.bar2 = bar
                self.b = nnx.Param(2)

        foo = Foo(Bar())

        def f(bar: Bar):
            bar.a += 1

        foo.for_each(Bar, f)

        assert foo.bar1.a == 2
        assert foo.bar2.a == 2

        def g(foo: Foo):
            foo.b += 1

        foo.for_each(Foo, g)

        assert foo.b == 3
