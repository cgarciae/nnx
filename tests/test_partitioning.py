import jax
import nnx
import typing as tp


def any_ref(x):
    return isinstance(x, nnx.Ref)


def has_collection(collection):
    return lambda x: isinstance(x, nnx.Referential) and x.collection == collection


class TestPartitioning:
    def test_partition_tree(self):
        p1 = nnx.Ref(1, "params")
        p2 = nnx.Ref(2, "params")
        s1 = nnx.Ref(3, "batch_stats")

        pytree = {
            "a": [p1, s1],
            "b": p2,
            "c": 100,
        }

        (params, rest), treedef = nnx.tree_partition(pytree, has_collection("params"))

        assert len(params) == 4
        assert len(rest) == 4

        # check params
        assert params[("a", "0")] is p1
        assert params[("a", "1")] is nnx.NOTHING
        assert params[("b",)] is p2
        assert params[("c",)] is nnx.NOTHING

        # check rest
        assert rest[("a", "0")] is nnx.NOTHING
        assert rest[("a", "1")] is s1
        assert rest[("b",)] is nnx.NOTHING
        assert rest[("c",)] == 100

        pytree = nnx.merge_partitions((params, rest), treedef)

        assert pytree["a"][0] is p1
        assert pytree["a"][1] is s1
        assert pytree["b"] is p2
        assert pytree["c"] == 100

    def test_update_from(self):
        p1 = nnx.Ref(1, "params")
        p2 = nnx.Ref(2, "params")
        s1 = nnx.Ref(3, "batch_stats")

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
        p1 = nnx.Ref(1.0, "params")
        s1 = nnx.Ref(-10, "batch_stats")

        pytree = {
            "a": [p1, s1],
            "b": p1,
            "c": 100,
        }

        params = nnx.get_partition(pytree, has_collection("params"))

        def loss(params, dagdef):
            params = nnx.reref(params, dagdef)
            return sum(p.value for p in jax.tree_util.tree_leaves(params))

        grad = jax.grad(loss)(*nnx.deref(params))
        nnx.update_refs(params, grad)

        assert pytree["a"][0].value == 2.0
        assert pytree["a"][1].value == -10
        assert pytree["b"].value == 2.0
        assert pytree["c"] == 100

    def test_get_paritition_idenpotent(self):
        p1 = nnx.Ref(10.0)
        p2 = nnx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [p1, p2],
            "b": p1,
            "c": 7,
            "d": 5.0,
        }

        ref_partition = nnx.get_partition(pytree, any_ref)
        assert ref_partition[("a", "0")] is p1
        assert ref_partition[("a", "1")] is p2
        assert ref_partition[("b",)] is p1
        assert ref_partition[("c",)] is nnx.NOTHING
        assert ref_partition[("d",)] is nnx.NOTHING
        assert len(ref_partition) == 5

        ref_partition = nnx.get_partition(ref_partition, any_ref)
        assert ref_partition[("a", "0")] is p1
        assert ref_partition[("a", "1")] is p2
        assert ref_partition[("b",)] is p1
        assert ref_partition[("c",)] is nnx.NOTHING
        assert ref_partition[("d",)] is nnx.NOTHING
        assert len(ref_partition) == 5
