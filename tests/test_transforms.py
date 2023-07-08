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
        m = nnx.Dict(a=nnx.Param(1))

        @nnx.jit
        def g(m: nnx.Dict):
            m.a = 2
            return 1.0

        out = g(m)

        assert m.a == 2
        assert out == 1.0

    def test_jit_stateless(self):
        m = nnx.Dict(a=nnx.Param(1))

        @partial(nnx.jit, stateful=False)
        def g(m: nnx.Dict):
            m.a = 2
            return 1.0

        out = g(m)

        assert m.a == 1
        assert out == 1.0


class TestGrad:
    def test_grad(self):
        p1 = nnx.Param(10.0)
        p2 = nnx.Param(20.0)

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
        assert grads["a/0"].value == 1.0
        assert isinstance(grads["a/0"], nnx.Node)
        assert grads["a/1"].value == 1.0
        assert isinstance(grads["a/1"], nnx.Node)
        assert grads["b"].value == 1.0
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
            a=nnx.Sequence([nnx.Param(10.0), nnx.BatchStat(20.0)]),
            b=nnx.Param(10.0),
            c=7,
            d=5.0,
        )

        @nnx.grad
        def f(m: nnx.Dict):
            # sum all params
            return m.a[0] + m.a[1] + m.b

        grads = f(m)

        assert isinstance(grads, nnx.State)
        assert grads["a/0"].value == 1.0
        assert isinstance(grads["a/0"], nnx.Param)
        assert len(grads) == 2

        m.update_state(grads)

        assert m.a[0] == 1.0
        assert m.a[1] == 20.0
        assert m.b == 1.0
        assert m.c == 7
        assert m.d == 5.0

    def test_grad_with_type_predicate(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.Param(10.0), nnx.BatchStat(20.0)]),
            b=nnx.Param(10.0),
            c=7,
            d=5.0,
        )

        @partial(nnx.grad, wrt=nnx.BatchStat)
        def f(m: nnx.Dict):
            # sum all params
            return m.a[0] + m.a[1] + m.b

        grads = f(m)

        assert isinstance(grads, nnx.State)
        assert grads["a/1"].value == 1.0
        assert isinstance(grads["a/1"], nnx.BatchStat)
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
            Block, variable_axes={nnx.Param: 0}, split_rngs="params", length=5
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
            variable_axes={nnx.Param: 0},
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

    def test_scan_with_sharding(self):
        class Block(nnx.Module):
            def __init__(self, *, ctx: nnx.Context):
                self.linear = nnx.Linear(
                    3,
                    3,
                    kernel_init=nnx.with_metadata(
                        nnx.initializers.lecun_normal(),
                        sharding=("din", "dout"),
                    ),
                    bias_init=nnx.with_metadata(
                        nnx.initializers.zeros(),
                        sharding=("dout",),
                    ),
                    ctx=ctx,
                )

            def __call__(self, x: jax.Array, _) -> tp.Tuple[jax.Array, None]:
                x = self.linear(x)

                # test sharding layer axes is not present inside scan
                state = self.linear.get_state()
                assert state["kernel"].value.shape == (3, 3)
                assert state["kernel"].sharding == ("din", "dout")
                assert state["bias"].value.shape == (3,)
                assert state["bias"].sharding == ("dout",)

                return x, None

        MLP = nnx.scan(
            Block,
            variable_axes={nnx.Param: 0},
            split_rngs=["params"],
            length=5,
            metadata_params={nnx.PARTITION_NAME: "layers"},
        )

        m = MLP(ctx=nnx.context(0))

        # test sharding layers axes is set
        state = m.get_state()
        assert state["scan_module/linear/kernel"].value.shape == (5, 3, 3)
        assert state["scan_module/linear/kernel"].sharding == ("layers", "din", "dout")
        assert state["scan_module/linear/bias"].value.shape == (5, 3)
        assert state["scan_module/linear/bias"].sharding == ("layers", "dout")

        x = jnp.ones((1, 3))
        y, out = m.call(x, None)

        # test sharding axes is preserved
        state = m.get_state()
        assert state["scan_module/linear/kernel"].value.shape == (5, 3, 3)
        assert state["scan_module/linear/kernel"].sharding == ("layers", "din", "dout")
        assert state["scan_module/linear/bias"].value.shape == (5, 3)
        assert state["scan_module/linear/bias"].sharding == ("layers", "dout")


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
            RematLinear, variable_axes={nnx.Param: 0}, split_rngs="params", length=5
        )

        m = ScanRematLinear(ctx=nnx.context(0))

        assert m.scan_module.remat_module.linear.kernel.shape == (5, 3, 3)
        assert m.scan_module.remat_module.linear.bias.shape == (5, 3)

        y, _ = m.call.call(jnp.ones((1, 3)), None)
        assert y.shape == (1, 3)

        y, _ = m(jnp.ones((1, 3)), None)
        assert y.shape == (1, 3)
