import typing as tp

import jax

import nnx


def any_ref(path, x):
    return isinstance(x, nnx.Variable)


def has_collection(collection):
    return lambda path, x: isinstance(x, nnx.Variable) and x.collection == collection


class TestPartitioning:
    def test_partition_tree(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(1), nnx.var("batch_stats", 2)]),
            b=nnx.param(2),
            c=100,
        )

        (params, rest), moduledef = m.partition("params", ...)

        assert len(params) == 2
        assert len(rest) == 1

        # check params
        assert params[("a", "0")].value == m.a[0]
        assert params[("b",)].value == m.b

        # check rest
        assert rest[("a", "1")].value == m.a[1]

        m2 = moduledef.merge(params, rest)

        assert m2.a[0] == m.a[0]
        assert m2.a[1] == m.a[1]
        assert m2.b == m.b
        assert m2.c == 100

    def test_update_from(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(1), nnx.var("batch_stats", 3)]),
            b=nnx.param(2),
            c=100,
        )

        state = m.partition()[0]
        state = jax.tree_map(lambda x: x * 2, state)

        m.update_state(state)

        assert m.a[0] == 2
        assert m.a[1] == 6
        assert m.b == 4
        assert m.c == 100

    def test_update_from_with_array_leaf(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(1), nnx.var("batch_stats", 3)]),
            b=nnx.param(2),
            c=jax.numpy.array(100),
        )

        pure_module: nnx.PureModule = m.partition()
        pure_module = jax.tree_map(lambda x: x * 2, pure_module)

        m.update_state(pure_module.state)

        assert m.a[0] == 2
        assert m.a[1] == 6
        assert m.b == 4
        assert m.c == 200

    def test_grad_example(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(1.0), nnx.var("batch_stats", -10)]),
            b=nnx.param(2.0),
            c=100,
        )

        params = m.filter("params")

        def loss(params):
            return sum(2 * p for p in jax.tree_util.tree_leaves(params))

        grads = jax.grad(loss)(params)
        m.update_state(grads)

        assert m.a[0] == 2.0
        assert m.a[1] == -10
        assert m.b == 2.0
        assert m.c == 100

    def test_get_paritition(self):
        m = nnx.Dict(
            a=nnx.Sequence([nnx.param(10.0), nnx.param(20.0)]),
            b=nnx.param(10.0),
            c=7,
            d=5.0,
        )

        # test Variables not shared
        assert vars(m.a)["0"] is not vars(m)["b"]

        state = m.filter(any_ref)
        assert state[("a", "0")].value == m.a[0]
        assert state[("a", "1")].value == m.a[1]
        assert state[("b",)].value == m.b
        assert state[("b",)] is not state[("a", "0")]
        assert len(state) == 3
