import dataclasses
import typing as tp
import jax

import refx

Predicate = tp.Callable[[tp.Any], bool]
CollectionFilter = tp.Union[str, tp.Sequence[str], Predicate]
LeafPredicate = tp.Callable[[tp.Any], bool]


def tree_partition(
    pytree,
    *filters: CollectionFilter,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> tp.Tuple[tp.Tuple[refx.Partition, ...], jax.tree_util.PyTreeDef]:
    predicates = (to_predicate(filter) for filter in filters)
    return refx.tree_partition(pytree, *predicates, is_leaf=is_leaf)


def get_partition(
    pytree,
    filter: CollectionFilter,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> refx.Partition:
    predicate = to_predicate(filter)
    return refx.get_partition(pytree, predicate, is_leaf=is_leaf)


def to_predicate(collection_filter: CollectionFilter) -> Predicate:
    if isinstance(collection_filter, str):
        return Is(collection_filter)
    elif isinstance(collection_filter, tp.Sequence):
        return In(collection_filter)
    elif callable(collection_filter):
        return collection_filter
    else:
        raise TypeError(f"Invalid collection filter: {collection_filter}")


merge_partitions = refx.merge_partitions


@dataclasses.dataclass
class Is:
    collection: str

    def __call__(self, x):
        return isinstance(x, refx.Referential) and x.collection == self.collection


@dataclasses.dataclass
class In:
    collections: tp.Sequence[str]

    def __post_init__(self):
        self.collections = tuple(self.collections)

    def __call__(self, x):
        return isinstance(x, refx.Referential) and x.collection in self.collections


class Not:
    def __init__(self, collection_filter: CollectionFilter):
        self.predicate = to_predicate(collection_filter)

    def __call__(self, x):
        return not self.predicate(x)
