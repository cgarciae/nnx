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

        pytree = {
            "a": [p1, s1],
            "b": p2,
            "c": 100,
        }

        (params, rest), dagdef = nnx.tree_partition(pytree, "params")

        assert len(params) == 4
        assert len(rest) == 4

        # check params
        assert params[("a", "0")].value == p1.value
        assert params[("a", "1")] is nnx.NOTHING
        assert params[("b",)].value == p2.value
        assert params[("c",)] is nnx.NOTHING

        # check rest
        assert rest[("a", "0")] is nnx.NOTHING
        assert rest[("a", "1")].value == s1.value
        assert rest[("b",)] is nnx.NOTHING
        assert rest[("c",)] == 100

        pytree = dagdef.reref((params, rest))

        assert pytree["a"][0].value == p1.value
        assert pytree["a"][1].value == s1.value
        assert pytree["b"].value == p2.value
        assert pytree["c"] == 100

    def test_update_from(self):
        p1 = nnx.Ref(1, collection="params")
        p2 = nnx.Ref(2, collection="params")
        s1 = nnx.Ref(3, collection="batch_stats")

        pytree = {
            "a": [p1, s1],
            "b": p2,
            "c": 100,
        }

        derered, dagdef = nnx.deref(pytree)
        derered = jax.tree_map(lambda x: x * 2, derered)

        nnx.update_refs(pytree, derered)

        assert pytree["a"][0].value == 2
        assert pytree["a"][1].value == 6
        assert pytree["b"].value == 4
        assert pytree["c"] == 100

    def test_grad_example(self):
        p1 = nnx.Ref(1.0, collection="params")
        s1 = nnx.Ref(-10, collection="batch_stats")

        pytree = {
            "a": [p1, s1],
            "b": p1,
            "c": 100,
        }

        params = nnx.get_partition(pytree, "params")

        def loss(params):
            return sum(2 * p for p in jax.tree_util.tree_leaves(params))

        grad = jax.grad(loss)(params)
        nnx.update_refs(pytree, grad)

        assert pytree["a"][0].value == 2.0
        assert pytree["a"][1].value == -10
        assert pytree["b"].value == 2.0
        assert pytree["c"] == 100

    def test_get_paritition(self):
        p1 = nnx.Ref(10.0)
        p2 = nnx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        partition = nnx.get_partition(pytree, any_ref)
        assert partition[("a", "0")].value == p1.value
        assert partition[("a", "1")].value == p2.value
        assert isinstance(partition[("b",)], nnx.Index)
        assert partition[("c",)] is nnx.NOTHING
        assert partition[("d",)] is nnx.NOTHING
        assert len(partition) == 5

    def test_nested_partition(self):
        p1 = nnx.Ref(10.0)
        p2 = nnx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
        }

        partition = nnx.get_partition(pytree, any_ref)
        assert partition[("a", "0")].value == p1.value
        assert partition[("a", "1")].value == p2.value
        assert len(partition) == 2

        pytree = {"x": partition}

        partition = nnx.get_partition(pytree, any_ref)
        assert ("x", "a", "0", "value") in partition
        assert ("x", "a", "1", "value") in partition
        assert len(partition) == 2
