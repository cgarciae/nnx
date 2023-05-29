import jax
import nnx
import typing as tp


def any_ref(path, x):
    return isinstance(x, nnx.Referential)


def has_collection(collection):
    return lambda path, x: isinstance(x, nnx.Referential) and x.collection == collection


class TestPartitioning:
    def test_partition_tree(self):
        p1 = nnx.Ref(1, collection="params")
        p2 = nnx.Ref(2, collection="params")
        s1 = nnx.Ref(3, collection="batch_stats")

        m = nnx.Map(
            a=nnx.Seq([p1, s1]),
            b=p2,
            c=100,
        )

        (params, rest), moddef = m.partition_general("params")

        assert len(params) == 2
        assert len(rest) == 1

        # check params
        assert params[("a", "0")].value == p1.value
        assert params[("b",)].value == p2.value

        # check rest
        assert rest[("a", "1")].value == s1.value

        m = moddef.reref((params, rest))

        assert m["a"][0].value == p1.value
        assert m["a"][1].value == s1.value
        assert m["b"].value == p2.value
        assert m["c"] == 100

    def test_update_from(self):
        p1 = nnx.Ref(1, collection="params")
        p2 = nnx.Ref(2, collection="params")
        s1 = nnx.Ref(3, collection="batch_stats")

        m = nnx.Map(
            a=nnx.Seq([p1, s1]),
            b=p2,
            c=100,
        )

        partition, moddef = m.deref()
        partition = jax.tree_map(lambda x: x * 2, partition)

        m.update(partition)

        assert m["a"][0].value == 2
        assert m["a"][1].value == 6
        assert m["b"].value == 4
        assert m["c"] == 100

    def test_update_from_with_array_leaf(self):
        p1 = nnx.Ref(1, collection="params")
        p2 = nnx.Ref(2, collection="params")
        s1 = nnx.Ref(3, collection="batch_stats")

        m = nnx.Map(
            a=nnx.Seq([p1, s1]),
            b=p2,
            c=jax.numpy.array(100),
        )

        dermod: nnx.DerefedMod = m.deref()
        dermod = jax.tree_map(lambda x: x * 2, dermod)

        m.update(dermod.partitions)

        assert m["a"][0].value == 2
        assert m["a"][1].value == 6
        assert m["b"].value == 4
        assert m["c"] == 200

    def test_grad_example(self):
        p1 = nnx.Ref(1.0, collection="params")
        s1 = nnx.Ref(-10, collection="batch_stats")

        m = nnx.Map(
            a=nnx.Seq([p1, s1]),
            b=p1,
            c=100,
        )

        params = m.get("params")

        def loss(params):
            return sum(2 * p for p in jax.tree_util.tree_leaves(params))

        grads = jax.grad(loss)(params)
        m.update(grads)

        assert m["a"][0].value == 2.0
        assert m["a"][1].value == -10
        assert m["b"].value == 2.0
        assert m["c"] == 100

    def test_get_paritition(self):
        p1 = nnx.Ref(10.0, "")
        p2 = nnx.Ref(20.0, "")

        m = nnx.Map(
            a=nnx.Seq([p1, p2]),
            b=p1,
            c=7,
            d=5.0,
        )

        partition = m.get(any_ref)
        assert partition[("a", "0")].value == p1.value
        assert partition[("a", "1")].value == p2.value
        assert isinstance(partition[("b",)], nnx.Index)
        assert len(partition) == 3
