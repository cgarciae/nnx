import dataclasses
import functools
import typing as tp
import jax
import jax.tree_util as jtu
from refx.ref import Ref, Deref, NOTHING, Nothing

from nnx.ref import DagDef, Referential, deref

A = tp.TypeVar("A")
Leaf = tp.Any
Leaves = tp.List[Leaf]
KeyPath = tp.Tuple[tp.Hashable, ...]


class StrPath(tp.Tuple[str, ...]):
    pass


##
Predicate = tp.Callable[[tp.Tuple[str, ...], tp.Any], bool]
CollectionFilter = tp.Union[
    str,
    tp.Sequence[str],
    Predicate,
]
LeafPredicate = tp.Callable[[tp.Any], bool]


class Partition(tp.Dict[tp.Tuple[str, ...], Leaf]):
    def __setitem__(self, key, value):
        raise TypeError("Partition is immutable")


def _partition_flatten_with_keys(
    x: Partition,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[StrPath, Leaf], ...], tp.Tuple[tp.Tuple[str, ...], ...]
]:
    children = tuple((StrPath(key), value) for key, value in x.items())
    return children, tuple(x.keys())


def _partition_unflatten(keys: tp.Tuple[StrPath, ...], leaves: tp.Tuple[Leaf, ...]):
    return Partition(zip(keys, leaves))


jax.tree_util.register_pytree_with_keys(
    Partition, _partition_flatten_with_keys, _partition_unflatten
)


def _key_path_to_str_gen(key_path: KeyPath) -> tp.Generator[str, None, None]:
    for key_entry in key_path:
        if isinstance(key_entry, StrPath):
            yield from key_entry
        elif isinstance(key_entry, jtu.SequenceKey):
            yield str(key_entry.idx)
        elif isinstance(key_entry, jtu.DictKey):  # "['a']"
            yield str(key_entry.key)
        elif isinstance(key_entry, jtu.GetAttrKey):
            yield str(key_entry.name)
        elif isinstance(key_entry, jtu.FlattenedIndexKey):
            yield str(key_entry.key)
        elif hasattr(key_entry, "__dict__") and len(key_entry.__dict__) == 1:
            yield str(next(iter(key_entry.__dict__.values())))
        else:
            yield str(key_entry)


def _to_str_path(key_path: KeyPath) -> StrPath:
    return StrPath(_key_path_to_str_gen(key_path))


class PartitionDef(tp.Generic[A]):
    __slots__ = ("_deref_dagdef", "_partition_treedef")

    def __init__(self, deref_dagdef: DagDef[A], partition_treedef: jtu.PyTreeDef):
        self._deref_dagdef = deref_dagdef
        self._partition_treedef = partition_treedef

    @property
    def deref_dagdef(self) -> DagDef[A]:
        return self._deref_dagdef

    @property
    def partition_treedef(self) -> jtu.PyTreeDef:
        return self._partition_treedef

    def __hash__(self) -> int:
        return hash((self._deref_dagdef, self._partition_treedef))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PartitionDef):
            raise TypeError(f"Cannot compare PartitionDef with {type(__value)}")

        return (
            self._deref_dagdef == __value._deref_dagdef
            and self._partition_treedef == __value._partition_treedef
        )

    def merge(self, partition: Partition, *partitions: Partition) -> A:
        partitions = (partition, *partitions)
        return merge_partitions(partitions, self)


def _partitiondef_flatten(
    x: PartitionDef[A],
) -> tp.Tuple[tp.Tuple[()], tp.Tuple[DagDef[A], jtu.PyTreeDef]]:
    return (), (x.deref_dagdef, x.partition_treedef)


def _partitiondef_unflatten(
    metadata: tp.Tuple[DagDef[A], jtu.PyTreeDef], _nodes
) -> PartitionDef[A]:
    return PartitionDef(*metadata)


jtu.register_pytree_node(PartitionDef, _partitiondef_flatten, _partitiondef_unflatten)


def tree_partition(
    pytree: A,
    *filters: CollectionFilter,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> tp.Tuple[tp.Tuple[Partition, ...], PartitionDef[A]]:
    if len(filters) == 0:
        raise ValueError("Expected at least one predicate")

    predicates = tuple(map(to_predicate, filters))

    pytree, dagdef = deref(pytree)

    paths_leaves, treedef = jax.tree_util.tree_flatten_with_path(
        pytree,
        is_leaf=lambda x: (isinstance(x, Deref) or x is NOTHING)
        or (False if is_leaf is None else is_leaf(x)),
    )

    partitiondef = PartitionDef(dagdef, treedef)

    paths_leaves = tuple((_to_str_path(path), leaf) for path, leaf in paths_leaves)
    paths = tuple(path for path, _leaf in paths_leaves)

    # we have n + 1 partitions, where n is the number of predicates
    # the last partition is for values that don't match any predicate
    partition_leaves: tp.Tuple[Leaves, ...] = tuple(
        [NOTHING] * len(paths_leaves) for _ in range(len(predicates) + 1)
    )
    for j, (path, leaf) in enumerate(paths_leaves):
        for i, predicate in enumerate(predicates):
            if predicate(path, leaf):
                partition_leaves[i][j] = leaf
                break
        else:
            # if we didn't break, set leaf to last partition
            partition_leaves[-1][j] = leaf

    partitions = tuple(
        Partition(zip(paths, partition)) for partition in partition_leaves
    )

    return partitions, partitiondef


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

    predicates = tuple(map(to_predicate, filters))

    (*partitions, _rest), _treedef = tree_partition(
        pytree, *predicates, is_leaf=is_leaf
    )

    assert len(partitions) == len(predicates)

    if len(partitions) == 1:
        partitions = partitions[0]
    else:
        partitions = tuple(partitions)

    return partitions


def _get_non_nothing(
    paths: tp.Tuple[tp.Tuple[str, ...], ...],
    leaves: tp.Tuple[tp.Union[Leaf, Nothing], ...],
    position: int,
):
    # check that all paths are the same
    paths_set = set(paths)
    if len(paths_set) != 1:
        raise ValueError(
            "All partitions must have the same paths, "
            f" at position [{position}] got "
            "".join(f"\n- {path}" for path in paths_set)
        )
    non_null = [option for option in leaves if option is not NOTHING]
    if len(non_null) == 0:
        raise ValueError(
            f"Expected at least one non-null value for position [{position}]"
        )
    elif len(non_null) > 1:
        raise ValueError(
            f"Expected at most one non-null value for position [{position}]"
        )
    return non_null[0]


def merge_partitions(
    partitions: tp.Sequence[Partition], partitiondef: PartitionDef[A]
) -> A:
    lenghts = [len(partition) for partition in partitions]
    if not all(length == lenghts[0] for length in lenghts):
        raise ValueError(
            "All partitions must have the same length, got "
            f"{', '.join(str(length) for length in lenghts)}"
        )

    partition_paths = (list(partition.keys()) for partition in partitions)
    partition_leaves = (list(partition.values()) for partition in partitions)

    merged_leaves = [
        _get_non_nothing(paths, leaves, i)
        for i, (paths, leaves) in enumerate(
            zip(zip(*partition_paths), zip(*partition_leaves))
        )
    ]

    pytree = jax.tree_util.tree_unflatten(partitiondef.partition_treedef, merged_leaves)
    return partitiondef.deref_dagdef.reref(pytree)


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
