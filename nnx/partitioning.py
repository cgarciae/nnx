import dataclasses
import functools
import typing as tp
import jax
import jax.tree_util as jtu

from nnx.ref import (
    NOTHING,
    DagDef,
    Deref,
    LeafPredicate,
    Partition,
    Referential,
    deref,
    _merge_partitions,
)

A = tp.TypeVar("A")
Leaf = tp.Any
Leaves = tp.List[Leaf]
KeyPath = tp.Tuple[tp.Hashable, ...]
Predicate = tp.Callable[[tp.Tuple[str, ...], tp.Any], bool]
CollectionFilter = tp.Union[
    str,
    tp.Sequence[str],
    Predicate,
]


def tree_partition(
    pytree: A,
    *filters: CollectionFilter,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> tp.Tuple[tp.Tuple[Partition, ...], DagDef[A]]:
    if len(filters) == 0:
        raise ValueError("Expected at least one predicate")

    predicates = tuple(map(to_predicate, filters))

    partition, dagdef = deref(pytree, is_leaf=is_leaf)

    # we have n + 1 partitions, where n is the number of predicates
    # the last partition is for values that don't match any predicate
    partition_leaves: tp.Tuple[Leaves, ...] = tuple(
        [NOTHING] * len(partition) for _ in range(len(predicates) + 1)
    )
    for j, (path, leaf) in enumerate(partition.items()):
        for i, predicate in enumerate(predicates):
            if predicate(path, leaf):
                partition_leaves[i][j] = leaf
                break
        else:
            # if we didn't break, set leaf to last partition
            partition_leaves[-1][j] = leaf

    paths = tuple(partition.keys())
    partitions = tuple(
        Partition(dict(zip(paths, partition))) for partition in partition_leaves
    )

    return partitions, dagdef


@tp.overload
def get_partition(
    pytree,
    filter: CollectionFilter,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> Partition:
    ...


@tp.overload
def get_partition(
    pytree,
    filter: CollectionFilter,
    *filters: CollectionFilter,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> tp.Tuple[Partition, ...]:
    ...


def get_partition(
    pytree,
    *filters: CollectionFilter,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> tp.Union[Partition, tp.Tuple[Partition, ...]]:
    if len(filters) == 0:
        raise ValueError("Expected at least one filter")

    (*partitions, _rest), _dagdef = tree_partition(pytree, *filters, is_leaf=is_leaf)

    assert len(partitions) == len(filters)

    if len(partitions) == 1:
        partitions = partitions[0]
    else:
        partitions = tuple(partitions)

    return partitions


def merge_partitions(partitions: tp.Sequence[Partition], dagdef: DagDef[A]) -> A:
    partition = _merge_partitions(partitions)
    return dagdef.reref(partition)


def to_predicate(collection_filter: CollectionFilter) -> Predicate:
    if isinstance(collection_filter, str):
        return Is(collection_filter)
    elif isinstance(collection_filter, tp.Sequence):
        return Any(collection_filter)
    elif callable(collection_filter):
        return collection_filter
    else:
        raise TypeError(f"Invalid collection filter: {collection_filter}")


@dataclasses.dataclass
class Is:
    collection: str

    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        if isinstance(x, Referential):
            return x.collection == self.collection
        return False


class Any:
    def __init__(self, collection_filters: tp.Sequence[CollectionFilter]):
        self.predicates = tuple(
            to_predicate(collection_filter) for collection_filter in collection_filters
        )

    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        return any(predicate(path, x) for predicate in self.predicates)


class Not:
    def __init__(self, collection_filter: CollectionFilter):
        self.predicate = to_predicate(collection_filter)

    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        return not self.predicate(path, x)
