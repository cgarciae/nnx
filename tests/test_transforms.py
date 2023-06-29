import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import pytest

import nnx


def collection(collection: str):
    return lambda x: isinstance(x, nnx.Node) and x.collection == collection


class TestJIT:
    def test_jit(self):
        m = nnx.Dict(a=nnx.param(1))

        @nnx.jit
        def g(m: nnx.Dict):
            m.a = 2
            return 1.0

        out = g(m)

        assert m.a == 2
        assert out == 1.0

    def test_jit_stateless(self):
        m = nnx.Dict(a=nnx.param(1))

        @partial(nnx.jit, stateful=False)
        def g(m: nnx.Dict):
            m.a = 2
            return 1.0

        out = g(m)

        assert m.a == 1
        assert out == 1.0


class TestGrad:
    def test_grad(self):
        p1 = nnx.param(10.0)
        p2 = nnx.param(20.0)

        m = nnx.Dict(
            a=nnx.Sequence([p1, p2]),
            b=p1,
            c=7,
            d=5.0,
        )

        @nnx.grad
        def f(m: nnx.Dict):
            # sum all params
            return m["a"][0] + m["a"][1] + m["b"]

        grads = f(m)

        assert isinstance(grads, nnx.State)
        # assert grads[("a", "0")].value == 1.0
        assert grads["a/0"].value == 1.0
        # assert isinstance(grads[("a", "0")], nnx.Variable)
        assert isinstance(grads["a/0"], nnx.Node)
        # assert grads[("a", "1")].value == 1.0
        assert grads["a/1"].value == 1.0
        # assert isinstance(grads[("a", "1")], nnx.Variable)
        assert isinstance(grads["a/1"], nnx.Node)
        # assert grads[("b",)].value == 1.0
        assert grads["b"].value == 1.0
        # assert isinstance(grads[("b",)], nnx.Variable)
        assert isinstance(grads["b"], nnx.Node)
        assert len(grads) == 3

        m.update_state(grads)

        assert m["a"][0] == 1.0
        assert m["a"][1] == 1.0
        assert m["b"] == 1.0
        assert m["c"] == 7
        assert m["d"] == 5.0

    def test_grad_with_multiple_ref_types(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(10.0), nnx.variable("batch_stats", 20.0)]),
            b=nnx.param(10.0),
            c=7,
            d=5.0,
        )

        @nnx.grad
        def f(m: nnx.Dict):
            # sum all params
            return m.a[0] + m.a[1] + m.b

        grads = f(m)

        assert isinstance(grads, nnx.State)
        # assert grads[("a", "0")].value == 1.0
        assert grads["a/0"].value == 1.0
        # assert isinstance(grads[("a", "0")], nnx.Variable)
        assert isinstance(grads["a/0"], nnx.Node)
        # assert grads[("a", "0")].collection == "params"
        assert grads["a/0"].collection == "params"
        assert len(grads) == 2

        m.update_state(grads)

        assert m.a[0] == 1.0
        assert m.a[1] == 20.0
        assert m.b == 1.0
        assert m.c == 7
        assert m.d == 5.0

    def test_grad_with_type_predicate(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(10.0), nnx.variable("batch_stats", 20.0)]),
            b=nnx.param(10.0),
            c=7,
            d=5.0,
        )

        @partial(nnx.grad, wrt="batch_stats")
        def f(m: nnx.Dict):
            # sum all params
            return m.a[0] + m.a[1] + m.b

        grads = f(m)

        assert isinstance(grads, nnx.State)
        # assert grads[("a", "1")].value == 1.0
        assert grads["a/1"].value == 1.0
        # assert isinstance(grads[("a", "1")], nnx.Variable)
        assert isinstance(grads["a/1"], nnx.Node)
        # assert grads[("a", "1")].collection == "batch_stats"
        assert grads["a/1"].collection == "batch_stats"
        assert len(grads) == 1

        m.update_state(grads)

        assert m.a[0] == 10.0
        assert m.a[1] == 1.0
        assert m.b == 10.0
        assert m.c == 7
        assert m.d == 5.0


class TestScan:
    def test_basic(self):
        class Block(nnx.Module):
            def __init__(self, *, ctx: nnx.Context):
                self.linear = nnx.Linear(3, 3, ctx=ctx)
                self.node = jnp.ones((2,))

            def __call__(self, x: jax.Array, _) -> tp.Tuple[jax.Array, None]:
                jax.debug.print("x={x}", x=x)
                x = self.linear(x)
                x = nnx.gelu(x)
                return x, None

        MLP = nnx.scan(
            Block, variable_axes={"params": 0}, split_rngs="params", length=5
        )

        module = MLP(ctx=nnx.context(0))

        assert module.scan_module.linear.kernel.shape == (5, 3, 3)
        assert module.scan_module.linear.bias.shape == (5, 3)
        assert module.scan_module.node.shape == (2,)

        x = jnp.ones((1, 3))
        y, out = module.call(x, None)

        assert y.shape == (1, 3)
        assert out is None

    def test_complex(self):
        class Block(nnx.Module):
            def __init__(self, *, ctx: nnx.Context):
                self.linear = nnx.Linear(3, 3, ctx=ctx)
                self.bn = nnx.BatchNorm(3, ctx=ctx)
                self.dropout = nnx.Dropout(0.5)
                self.node = jnp.ones((2,))

            def __call__(
                self, x: jax.Array, _, *, ctx: nnx.Context
            ) -> tp.Tuple[jax.Array, None]:
                jax.debug.print("x={x}", x=x)
                x = self.linear(x)
                x = self.bn(x, ctx=ctx)
                x = self.dropout(x, ctx=ctx)
                x = nnx.gelu(x)
                return x, None

        MLP = nnx.scan(
            Block,
            variable_axes={"params": 0},
            # variable_carry="batch_stats",
            split_rngs=["params", "dropout"],
            length=5,
        )

        module = MLP(ctx=nnx.context(0))

        assert module.scan_module.linear.kernel.shape == (5, 3, 3)
        assert module.scan_module.linear.bias.shape == (5, 3)
        assert module.scan_module.node.shape == (2,)

        x = jnp.ones((1, 3))
        ctx = nnx.context(
            dropout=1, flags=dict(deterministic=False, use_running_average=False)
        )
        y, out = module.call(x, None, ctx=ctx)

        assert y.shape == (1, 3)
        assert out is None

    def test_add_metadata_axis(self):
        return
        state_copy = None

        class Foo(nnx.Module):
            def __init__(self, *, ctx: nnx.Context):
                kernel_init = nnx.with_partitioning(
                    nnx.initializers.lecun_normal(), ("foo", "bar")
                )
                self.linear = nnx.Linear(
                    4, 4, kernel_init=kernel_init, use_bias=False, ctx=ctx
                )

            @nn.compact
            def __call__(self, x):
                nonlocal state_copy
                state_copy = self.get_state()
                return self.linear(x)

        class Test(nnx.Module):
            @partial(
                nn.add_metadata_axis,
                variable_axes={"params": 0},
                metadata_params={nn.PARTITION_NAME: "baz"},
            )
            @nn.compact
            def __call__(self, x):
                return Foo(name="foo")(x)

        k = random.PRNGKey(0)
        x = jnp.ones((4, 4), dtype=jnp.float32)
        vs = Test().init(k, x)
        y = Test().apply(vs, x)
        outer_expect = jax.tree_map(
            jnp.shape,
            freeze(
                {
                    "params": {
                        "foo": {
                            "dense": {
                                "kernel": nn.Partitioned(
                                    jnp.ones((4, 4)), names=("baz", "foo", "bar")
                                )
                            }
                        }
                    }
                }
            ),
        )
        inner_expect = jax.tree_map(
            jnp.shape,
            freeze(
                {
                    "params": {
                        "dense": {
                            "kernel": nn.Partitioned(
                                jnp.ones((4, 4)), names=("foo", "bar")
                            )
                        }
                    }
                }
            ),
        )
        self.assertEqual(jax.tree_map(jnp.shape, vs), outer_expect)
        self.assertEqual(jax.tree_map(jnp.shape, state_copy), inner_expect)


class TestRemat:
    def test_basic_remat(self):
        RematLinear = nnx.remat(nnx.Linear)

        module = RematLinear(2, 3, ctx=nnx.context(0))

        y = module.call(jnp.ones((1, 2)))

        assert y.shape == (1, 3)

    def test_remat_with_scan(self):
        class LinearBlock(nnx.Module):
            def __init__(self, *, ctx: nnx.Context):
                self.linear = nnx.Linear(3, 3, ctx=ctx)

            def __call__(self, x: jax.Array, _) -> tp.Tuple[jax.Array, None]:
                x = self.linear(x)
                return x, None

        RematLinear = nnx.remat(LinearBlock)

        ScanRematLinear = nnx.scan(
            RematLinear, variable_axes={"params": 0}, split_rngs="params", length=5
        )

        m = ScanRematLinear(ctx=nnx.context(0))

        assert m.scan_module.remat_module.linear.kernel.shape == (5, 3, 3)
        assert m.scan_module.remat_module.linear.bias.shape == (5, 3)

        y, _ = m.call.call(jnp.ones((1, 3)), None)
        assert y.shape == (1, 3)

        y, _ = m(jnp.ones((1, 3)), None)
        assert y.shape == (1, 3)
