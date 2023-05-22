from abc import ABC, abstractmethod
from functools import partial
from types import MappingProxyType
import typing as tp
import nnx

import jax
import jax.tree_util as jtu

from nnx import tracers

A = tp.TypeVar("A")
D = tp.TypeVar("D", bound="DagDef[tp.Any]")
Leaf = tp.Any
Leaves = tp.List[Leaf]
DagIndexes = tp.Tuple[tp.Tuple[int, ...], ...]
DagIndexesList = tp.List[tp.List[int]]
LeafPredicate = tp.Callable[[tp.Any], bool]
KeyPath = tp.Tuple[tp.Hashable, ...]


class StrPath(tp.Tuple[str, ...]):
    pass


class Partition(tp.Mapping[tp.Tuple[str, ...], Leaf]):
    @tp.overload
    def __init__(self, __mapping: tp.Mapping[tp.Tuple[str, ...], Leaf], /):
        ...

    @tp.overload
    def __init__(self, __iterable: tp.Iterable[tp.Tuple[tp.Tuple[str, ...], Leaf]], /):
        ...

    def __init__(
        self,
        __input: tp.Union[
            tp.Mapping[tp.Tuple[str, ...], Leaf],
            tp.Iterable[tp.Tuple[tp.Tuple[str, ...], Leaf]],
        ],
        /,
    ):
        if not isinstance(__input, tp.Mapping):
            __input = dict(__input)
        self._mapping = MappingProxyType(__input)

    def __getitem__(self, __key: tp.Tuple[str, ...]) -> Leaf:
        return self._mapping[__key]

    def __iter__(self) -> tp.Iterator[tp.Tuple[str, ...]]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)


def _partition_flatten_with_keys(
    x: Partition,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[StrPath, Leaf], ...], tp.Tuple[tp.Tuple[str, ...], ...]
]:
    children = tuple((StrPath(key), value) for key, value in x.items())
    return children, tuple(x.keys())


def _partition_unflatten(keys: tp.Tuple[StrPath, ...], leaves: tp.Tuple[Leaf, ...]):
    return Partition(dict(zip(keys, leaves)))


jax.tree_util.register_pytree_with_keys(
    Partition, _partition_flatten_with_keys, _partition_unflatten
)


def _to_str_path_gen(key_path: KeyPath) -> tp.Iterator[str]:
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


def _remove_last_ref(iterator: tp.Iterator[str]) -> tp.Iterator[str]:
    last: tp.Optional[str] = None
    for key in iterator:
        if isinstance(last, str):
            yield last
        last = key

    if isinstance(last, str):
        if last.endswith("__ref"):
            last = last[:-5]
        yield last


def _to_str_path(key_path: KeyPath) -> tp.Tuple[str, ...]:
    return tuple(_remove_last_ref(_to_str_path_gen(key_path)))


class DagDef(tp.Generic[A]):
    __slots__ = ("_indexes", "_treedef")

    def __init__(self, indexes: DagIndexes, treedef: jtu.PyTreeDef):
        self._indexes = indexes
        self._treedef = treedef

    def unflatten(self, leaves: Leaves) -> A:
        return self._treedef.unflatten(leaves)

    def flatten_up_to(self, __pytree: tp.Any, /) -> Leaves:
        return self.treedef.flatten_up_to(__pytree)

    @property
    def indexes(self) -> DagIndexes:
        return self._indexes

    @property
    def treedef(self) -> jtu.PyTreeDef:
        return self._treedef

    def __hash__(self) -> int:
        return hash((self._indexes, self._treedef))

    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, DagDef):
            raise TypeError(f"Cannot compare DagDef with {type(other).__name__}")
        return self._indexes == other._indexes and self._treedef == other._treedef

    def reref(self, partition: Partition) -> A:
        return reref(partition, self)

    def merge(
        self,
        partitions: tp.Union[tp.Sequence[Partition], tp.Mapping[str, Partition]],
    ) -> A:
        if isinstance(partitions, tp.Mapping):
            partitions = tuple(partitions.values())
        return nnx.merge(partitions, self)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        jtu.register_pytree_node(
            cls, _dagdef_flatten, partial(_dagdef_unflatten, cls=cls)
        )


def _dagdef_flatten(
    x: DagDef[A],
) -> tp.Tuple[tp.Tuple[()], tp.Tuple[DagIndexes, jtu.PyTreeDef]]:
    return (), (x._indexes, x._treedef)


def _dagdef_unflatten(
    metadata: tp.Tuple[DagIndexes, jtu.PyTreeDef], _: tp.Tuple[()], *, cls: tp.Type[D]
) -> D:
    return cls(*metadata)


jtu.register_pytree_node(
    DagDef, _dagdef_flatten, partial(_dagdef_unflatten, cls=DagDef)
)


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"  # pragma: no cover


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jtu.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


class Referential(tp.Generic[A], ABC):
    __slots__ = ("_collection",)

    def __init__(self, collection: str):
        self._collection = collection

    @property
    @abstractmethod
    def value(self) -> A:
        ...

    @property
    def collection(self) -> str:
        return self._collection


class Deref(Referential[A]):
    __slots__ = ()


class Ref(Referential[A]):
    __slots__ = ("_value", "_jax_trace", "_context_trace", "_trace_set")

    def __init__(
        self,
        value: A,
        *,
        collection: str = "",
        context_trace: tp.Optional[tracers.MainTrace] = None,
    ):
        self._value = value
        self._jax_trace = tracers.current_jax_trace()
        self._context_trace = context_trace or self._jax_trace
        self._trace_set = frozenset((self._jax_trace, self._context_trace))
        super().__init__(collection)

    @property
    def value(self) -> A:
        # TODO: passing references as a constant to a function as a capture should be allowed
        # maybe access should always be allowed? Commenting out for now.

        # value_trace = tracers.get_top_trace(self._value)
        # if self._jax_trace is not tracers.current_jax_trace() or (
        #     value_trace is not self._jax_trace
        #     and value_trace is not self._context_trace
        # ):
        #     raise ValueError("Cannot access ref from different trace level")
        return self._value

    @value.setter
    def value(self, value: A):
        value_trace = tracers.get_top_trace(self._value)
        if self._jax_trace is not tracers.current_jax_trace() or (
            value_trace is not self._jax_trace
            and value_trace is not self._context_trace
        ):
            raise ValueError("Cannot mutate ref from different trace level")

        invalid_traces = tracers.get_all_traces(value) - self._trace_set
        if invalid_traces:
            raise ValueError(
                "Cannot mutate ref with value that contains tracers from other "
                f"traces: {invalid_traces}"
            )

        self._value = value

    def to_value(self) -> "Value[A]":
        return Value(self.value, self.collection)

    def to_index(self) -> "Index[A]":
        return Index(self.collection)


class Value(Deref[A]):
    __slots__ = ("_value",)

    def __init__(self, value: A, collection: str):
        self._value = value
        super().__init__(collection)

    @property
    def value(self) -> A:
        return self._value

    def to_ref(self) -> "Ref[A]":
        return Ref(self._value, collection=self.collection)

    def __repr__(self) -> str:
        return f"Value(collection={repr(self.collection)}, value={repr(self._value)})"


def _value_flatten(
    x: Value[A],
    *,
    with_keys: bool,
) -> tp.Tuple[tp.Tuple[tp.Any], tp.Hashable]:
    if with_keys:
        node = (jtu.GetAttrKey("value"), x._value)
    else:
        node = x._value

    return (node,), x.collection


def _value_unflatten(collection: tp.Hashable, children: tp.Tuple[A]) -> Value[A]:
    return Value(children[0], collection)


jtu.register_pytree_with_keys(
    Value,
    partial(_value_flatten, with_keys=True),
    _value_unflatten,
    flatten_func=partial(_value_flatten, with_keys=False),
)


class Index(Deref[A]):
    __slots__ = ()

    def __init__(self, collection: tp.Hashable):
        self._collection = collection

    @property
    def value(self) -> A:
        raise ValueError(f"Cannot get value from '{type(self).__name__}' instances")

    def __repr__(self) -> str:
        return f"Index(collection={self.collection})"


def _index_flatten(x: Index[A]) -> tp.Tuple[tp.Tuple[()], tp.Hashable]:
    return (), x.collection


def _index_unflatten(colletion: tp.Hashable, children: tp.Tuple[()]) -> Index[A]:
    return Index(colletion)


jtu.register_pytree_node(Index, _index_flatten, _index_unflatten)


class Dag(tp.Generic[A]):
    __slots__ = ("_value",)

    def __init__(self, value: A):
        self._value = value

    @property
    def value(self) -> A:
        return self._value


def _dag_flatten(
    x: Dag[A],
    *,
    with_keys: bool,
) -> tp.Tuple[tp.Tuple[tp.Any], DagDef[A]]:
    leaves, dagdef = deref(x.value)
    if with_keys:
        node = (jtu.GetAttrKey("value"), leaves)
    else:
        node = leaves
    return (node,), dagdef


def _dag_unflatten(dagdef: DagDef[A], nodes: tp.Tuple[Leaves]) -> Dag[A]:
    return Dag(reref(nodes[0], dagdef))


jtu.register_pytree_with_keys(
    Dag,
    partial(_dag_flatten, with_keys=True),
    _dag_unflatten,
    flatten_func=partial(_dag_flatten, with_keys=False),
)


@tp.overload
def deref(
    pytree: A,
    *,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> tp.Tuple[Partition, DagDef[A]]:
    ...


@tp.overload
def deref(
    pytree: A,
    *,
    is_leaf: tp.Optional[LeafPredicate] = None,
    unflatten: tp.Literal[False],
) -> tp.Tuple[Partition, DagDef[A]]:
    ...


@tp.overload
def deref(
    pytree: A,
    *,
    is_leaf: tp.Optional[LeafPredicate] = None,
    unflatten: tp.Literal[True],
) -> tp.Tuple[A, DagDef[A]]:
    ...


@tp.overload
def deref(
    pytree: A,
    *,
    is_leaf: tp.Optional[LeafPredicate] = None,
    unflatten: bool,
) -> tp.Tuple[tp.Union[Partition, A], DagDef[A]]:
    ...


def deref(
    pytree: A,
    *,
    is_leaf: tp.Optional[LeafPredicate] = None,
    unflatten: bool = False,
) -> tp.Tuple[tp.Union[Partition, A], DagDef[A]]:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}
    indexes: tp.List[tp.List[int]] = []

    if unflatten:
        leaves, treedef = jtu.tree_flatten(pytree, is_leaf=is_leaf)
    else:
        leaves, treedef = jtu.tree_flatten_with_path(pytree, is_leaf=is_leaf)

    out_leaves: Leaves = []
    out_paths: tp.List[tp.Tuple[str, ...]] = []

    for i, leaf in enumerate(leaves):
        if unflatten:
            path, x = None, leaf
        else:
            path, x = leaf

        if isinstance(x, Ref):
            if x not in ref_index:
                ref_index[x] = len(ref_index)
                indexes.append([i])
                x = x.to_value()
            else:
                indexes[ref_index[x]].append(i)
                x = x.to_index()
        elif isinstance(x, Deref):
            raise ValueError("Cannot 'deref' pytree containing Derefs")

        out_leaves.append(x)
        if path is not None:
            path = StrPath(_remove_last_ref(_to_str_path_gen(path)))
            out_paths.append(path)

    _indexes = tuple(map(tuple, indexes))
    dagdef = DagDef[A](_indexes, treedef)

    if unflatten:
        out = dagdef.unflatten(out_leaves)
    else:
        out = Partition(dict(zip(out_paths, out_leaves)))

    return out, dagdef


def reref(__tree_or_partition: tp.Union[Partition, A], /, dagdef: DagDef[A]) -> A:
    if isinstance(__tree_or_partition, Partition):
        leaves = list(__tree_or_partition.values())
    else:
        leaves = dagdef.flatten_up_to(__tree_or_partition)
    context_trace = tracers.get_top_trace(leaves)

    for leaf_indexes in dagdef.indexes:
        leaf_index = leaf_indexes[0]
        value = leaves[leaf_index]

        if not isinstance(value, Value):
            raise ValueError(
                f"Expected 'Value' as first leaf, got {type(value).__name__}"
            )

        ref = Ref(value.value, collection=value.collection, context_trace=context_trace)
        leaves[leaf_index] = ref

        for leaf_index in leaf_indexes[1:]:
            x = leaves[leaf_index]

            if not isinstance(x, Index):
                raise ValueError(f"Expected 'Index' as leaf, got {type(x).__name__}")

            leaves[leaf_index] = ref

    return dagdef.unflatten(leaves)


def clone(pytree: A) -> A:
    return reref(*deref(pytree))


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
    non_nothing = [option for option in leaves if option is not NOTHING]
    if len(non_nothing) == 0:
        raise ValueError(
            f"Expected at least one non-null value for position [{position}]"
        )
    elif len(non_nothing) > 1:
        raise ValueError(
            f"Expected at most one non-null value for position [{position}]"
        )
    return non_nothing[0]


def _merge_partitions(partitions: tp.Sequence[Partition]) -> Partition:
    if len(partitions) == 0:
        raise ValueError("Expected at least one partition")

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

    return Partition(dict(zip(partitions[0].keys(), merged_leaves)))


def update_refs(pytree: tp.Any, partition: Partition, *partitions: Partition):
    if len(partitions) > 0:
        partition = _merge_partitions((partition, *partitions))
    target_leaves = jtu.tree_leaves(pytree)
    source_leaves = list(partition.values())
    _update_refs(target_leaves, source_leaves)


def _update_refs(target_leaves: tp.List[Leaf], source_leaves: tp.List[Leaf]):
    if len(target_leaves) != len(source_leaves):
        raise ValueError(
            f"Target and source leaves must have the same length, got "
            f"{len(target_leaves)} and {len(source_leaves)}"
        )

    seen_target_refs: tp.Set[Ref[tp.Any]] = set()

    for i, (target_leaf, source_leaf) in enumerate(zip(target_leaves, source_leaves)):
        if isinstance(source_leaf, (Value, Index)):
            if not isinstance(target_leaf, Ref):
                raise ValueError(
                    f"Expected 'Ref' as target leaf at position [{i}], "
                    f"got {type(target_leaf).__name__}"
                )
            elif isinstance(source_leaf, Index):
                if target_leaf not in seen_target_refs:
                    raise ValueError(
                        f"Found source 'Index' at position [{i}] but 'Ref' "
                        f"has not been seen before."
                    )
            else:
                if target_leaf in seen_target_refs:
                    raise ValueError(
                        f"Found source '{type(source_leaf).__name__}' at position [{i}] "
                        f"but corresponding 'Ref' has been seen before."
                    )
                target_leaf.value = source_leaf.value
                seen_target_refs.add(target_leaf)
