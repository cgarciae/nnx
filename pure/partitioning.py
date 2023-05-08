import typing as tp
import jax
import jax.tree_util as jtu
from pure.state import Variable

A = tp.TypeVar("A")
CollectionPredicate = tp.Callable[[str], bool]
Leaf = tp.Any
Leaves = tp.List[Leaf]
KeyPath = tp.Tuple[tp.Hashable, ...]
LeafPredicate = tp.Callable[[tp.Any], bool]


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"  # pragma: no cover


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jtu.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


class StrPath(tp.Tuple[str, ...]):
    pass


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


def _key_path_to_str_path(key_path: KeyPath) -> StrPath:
    return StrPath(_key_path_to_str_gen(key_path))


def tree_partition(
    pytree,
    *predicates: CollectionPredicate,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> tp.Tuple[tp.Tuple[Partition, ...], jax.tree_util.PyTreeDef]:
    paths_leaves: tp.List[tp.Tuple[KeyPath, Leaf]]
    paths_leaves, treedef = jax.tree_util.tree_flatten_with_path(
        pytree,
        is_leaf=lambda x: (isinstance(x, Variable) or x is NOTHING)
        or (False if is_leaf is None else is_leaf(x)),
    )

    leaves: tp.Tuple[Leaf, ...]
    paths, leaves = zip(*paths_leaves)
    paths = tuple(map(_key_path_to_str_path, paths))

    # we have n + 1 partitions, where n is the number of predicates
    # the last partition is for values that don't match any predicate
    partition_leaves: tp.Tuple[Leaves, ...] = tuple(
        [NOTHING] * len(leaves) for _ in range(len(predicates) + 1)
    )
    for j, leaf in enumerate(leaves):
        for i, predicate in enumerate(predicates):
            if isinstance(leaf, Variable) and predicate(leaf.collection):
                partition_leaves[i][j] = leaf
                break
        else:
            # if we didn't break, set leaf to last partition
            partition_leaves[-1][j] = leaf

    partitions = tuple(
        Partition(zip(paths, partition)) for partition in partition_leaves
    )
    return partitions, treedef


def get_partition(
    pytree,
    predicate: CollectionPredicate,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> Partition:
    (partition, _rest), _treedef = tree_partition(pytree, predicate, is_leaf=is_leaf)
    return partition


def _get_non_nothing(
    paths: tp.Tuple[StrPath, ...],
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
    partitions: tp.Sequence[Partition], treedef: jax.tree_util.PyTreeDef
):
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

    return jax.tree_util.tree_unflatten(treedef, merged_leaves)
