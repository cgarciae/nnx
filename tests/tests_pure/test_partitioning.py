import jax
import pure
import typing as tp


def any_collection(x):
    return True


def has_collection(collection: str):
    return lambda x: x == collection


class TestPartitioning:
    def test_partition_tree(self):
        p1 = pure.Variable(1, "params")
        p2 = pure.Variable(2, "params")
        s1 = pure.Variable(3, "batch_stats")

        pytree = {
            "a": [p1, s1],
            "b": p2,
            "c": 100,
        }

        (params, rest), treedef = pure.tree_partition(pytree, has_collection("params"))

        assert len(params) == 4
        assert len(rest) == 4

        # check params
        assert params[("a", "0")] is p1
        assert params[("a", "1")] is pure.NOTHING
        assert params[("b",)] is p2
        assert params[("c",)] is pure.NOTHING

        # check rest
        assert rest[("a", "0")] is pure.NOTHING
        assert rest[("a", "1")] is s1
        assert rest[("b",)] is pure.NOTHING
        assert rest[("c",)] == 100

        pytree = pure.merge_partitions((params, rest), treedef)

        assert pytree["a"][0] is p1
        assert pytree["a"][1] is s1
        assert pytree["b"] is p2
        assert pytree["c"] == 100

    def test_grad_example(self):
        p1 = pure.Variable(1.0, "params")
        s1 = pure.Variable(-10, "batch_stats")

        pytree = {
            "a": [p1, s1],
            "b": p1,
            "c": 100,
        }

        def loss(params, rest, treedef):
            params = pure.merge_partitions([params, rest], treedef)
            return sum(jax.tree_util.tree_leaves(params))

        (params, rest), treedef = pure.tree_partition(pytree, has_collection("params"))

        grads = jax.grad(loss)(params, rest, treedef)

        pytree = pure.merge_partitions([grads, rest], treedef)

        assert pytree["a"][0].value == 1.0
        assert pytree["a"][1].value == -10
        assert pytree["b"].value == 1.0
        assert pytree["c"] == 100

    def test_get_paritition_idenpotent(self):
        p1 = pure.Variable(10.0)
        p2 = pure.Variable(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        ref_partition = pure.get_partition(pytree, any_collection)
        assert ref_partition[("a", "0")] is p1
        assert ref_partition[("a", "1")] is p2
        assert ref_partition[("b",)] is p1
        assert ref_partition[("c",)] is pure.NOTHING
        assert ref_partition[("d",)] is pure.NOTHING
        assert len(ref_partition) == 5

        ref_partition = pure.get_partition(ref_partition, any_collection)
        assert ref_partition[("a", "0")] is p1
        assert ref_partition[("a", "1")] is p2
        assert ref_partition[("b",)] is p1
        assert ref_partition[("c",)] is pure.NOTHING
        assert ref_partition[("d",)] is pure.NOTHING
        assert len(ref_partition) == 5
