import dataclasses
import functools
import typing as tp
import jax

import refx

Predicate = tp.Callable[[tp.Any], bool]
CollectionPredicate = tp.Callable[[tp.Hashable], bool]
CollectionFilter = tp.Union[
    str, tp.Sequence[str], CollectionPredicate, tp.Sequence[CollectionPredicate]
]
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


def filter_collection(
    f: tp.Callable[[CollectionFilter], CollectionPredicate]
) -> tp.Callable[[CollectionFilter], Predicate]:
    @functools.wraps(f)
    def wrapper(collection_filter: CollectionFilter) -> Predicate:
        collection_predicate = f(collection_filter)
        return lambda x: isinstance(x, refx.Referential) and collection_predicate(
            x.collection
        )

    return wrapper


@filter_collection
def to_predicate(collection_filter: CollectionFilter) -> CollectionPredicate:
    if isinstance(collection_filter, str):
        return Is(collection_filter)
    elif isinstance(collection_filter, tp.Sequence):
        return Any(collection_filter)
    elif callable(collection_filter):
        return collection_filter
    else:
        raise TypeError(f"Invalid collection filter: {collection_filter}")


merge_partitions = refx.merge_partitions


@dataclasses.dataclass
class Is:
    collection: str

    def __call__(self, collection):
        return collection == self.collection


class Any:
    def __init__(self, collection_filters: tp.Sequence[CollectionFilter]):
        self.predicates = tuple(
            to_predicate(collection_filter) for collection_filter in collection_filters
        )

    def __call__(self, collection):
        return any(predicate(collection) for predicate in self.predicates)


class Not:
    def __init__(self, collection_filter: CollectionFilter):
        self.predicate = to_predicate(collection_filter)

    def __call__(self, x):
        return not self.predicate(x)
