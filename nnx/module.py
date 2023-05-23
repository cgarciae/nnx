import dataclasses
from nnx.pytree import Pytree
from nnx import partitioning
from nnx.reference import (
    DagDef,
    Partition,
    Referential,
    clone,
    deref,
    reref,
    update_refs,
    _to_str_path,
)
import typing as tp
import jax.tree_util as jtu
import builtins


M = tp.TypeVar("M", bound="Module")

if tp.TYPE_CHECKING:
    SetItemType = tp.Union[builtins.ellipsis, builtins.slice]
else:
    SetItemType = tp.Any


class _ProxyContext(tp.Protocol):
    def __call__(self, __fn: tp.Callable[..., tp.Any], *args, **kwargs) -> tp.Any:
        ...


@dataclasses.dataclass
class CallableProxy:
    _proxy_context: _ProxyContext
    _proxy_callable: tp.Callable[..., tp.Any]

    def __call__(self, *args, **kwargs):
        return self._proxy_context(self._proxy_callable, *args, **kwargs)

    def __getattr__(self, name) -> "CallableProxy":
        return CallableProxy(self._proxy_context, getattr(self._proxy_callable, name))


class Module(Pytree):
    @tp.overload
    def deref(self: M) -> tp.Tuple[Partition, DagDef[M]]:
        ...

    @tp.overload
    def deref(
        self: M, *, unflatten: tp.Literal[False]
    ) -> tp.Tuple[Partition, DagDef[M]]:
        ...

    @tp.overload
    def deref(self: M, *, unflatten: tp.Literal[True]) -> tp.Tuple[M, DagDef[M]]:
        ...

    @tp.overload
    def deref(
        self: M, *, unflatten: bool
    ) -> tp.Tuple[tp.Union[M, Partition], DagDef[M]]:
        ...

    def deref(
        self: M, *, unflatten: bool = False
    ) -> tp.Tuple[tp.Union[M, Partition], DagDef[M]]:
        return deref(self, unflatten=unflatten)

    def reref(self: M, dagdef: DagDef[M]) -> M:
        return reref(self, dagdef)

    def clone(self: M) -> M:
        return clone(self)

    @tp.overload
    def __getitem__(self, collection: str) -> Partition:
        ...

    @tp.overload
    def __getitem__(self, collections: tp.Tuple[str, ...]) -> tp.Tuple[Partition, ...]:
        ...

    def __getitem__(
        self, collections: tp.Union[str, tp.Tuple[str, ...]]
    ) -> tp.Union[Partition, tp.Tuple[Partition, ...]]:
        if len(collections) < 1:
            raise ValueError("Must specify at least one collection")

        if isinstance(collections, str):
            collections = (collections,)

        return partitioning.get_partition(self, *collections)

    def __setitem__(self, name: SetItemType, value: Partition):
        update_refs(self, value)

    def collections(self) -> tp.FrozenSet[str]:
        return frozenset(
            x.collection
            for x in jtu.tree_leaves(self, is_leaf=lambda x: isinstance(x, Referential))
            if isinstance(x, Referential)
        )

    @tp.overload
    def partition(
        self: M,
        *,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> "Bounded[M]":
        ...

    @tp.overload
    def partition(
        self: M,
        collection: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Tuple[Partition, "ModuleDef[M]",]:
        ...

    @tp.overload
    def partition(
        self: M,
        collection: str,
        *collections: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Tuple[tp.Tuple[Partition, ...], "ModuleDef[M]",]:
        ...

    def partition(
        self: M,
        *collections: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Union[
        "Bounded[M]",
        tp.Tuple[
            tp.Union[tp.Tuple[Partition, ...], Partition],
            "ModuleDef[M]",
        ],
    ]:
        if len(collections) == 0:
            partitions, dagdef = partitioning.collection_partition(
                self, is_leaf=is_leaf
            )
        elif len(collections) == 1:
            partitions, dagdef = partitioning.collection_partition(
                self, collections[0], is_leaf=is_leaf
            )
        else:
            partitions, dagdef = partitioning.collection_partition(
                self, collections[0], *collections[1:], is_leaf=is_leaf
            )

        moduledef = ModuleDef(dagdef.indexes, dagdef.treedef)

        if isinstance(partitions, tp.Dict):
            return Bounded(partitions, moduledef)

        return partitions, moduledef

    @tp.overload
    def ref_dict(self) -> tp.Tuple[tp.Dict[tp.Tuple[str, ...], tp.Any], jtu.PyTreeDef]:
        ...

    @tp.overload
    def ref_dict(self, sep: str) -> tp.Tuple[tp.Dict[str, tp.Any], jtu.PyTreeDef]:
        ...

    def ref_dict(
        self, sep: tp.Optional[str] = None
    ) -> tp.Tuple[
        tp.Union[tp.Dict[tp.Tuple[str, ...], tp.Any], tp.Dict[str, tp.Any]],
        jtu.PyTreeDef,
    ]:
        path_leaves, treedef = jtu.tree_flatten_with_path(self)
        partition = ((_to_str_path(path), leaf) for path, leaf in path_leaves)
        partition = (
            (path[-1][:-5] if path[-1].endswith("__ref") else path, leaf)
            for path, leaf in partition
        )
        if sep is None:
            partition = {_to_str_path(path): leaf for path, leaf in path_leaves}
        else:
            partition = {
                sep.join(_to_str_path(path)): leaf for path, leaf in path_leaves
            }
        return partition, treedef


class ApplyCaller(tp.Protocol):
    def __getattr__(self, __name) -> "ApplyCaller":
        ...

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, tp.Dict[str, Partition]]:
        ...


class ModuleDef(DagDef[M]):
    def apply(
        self, partitions: tp.Union[tp.Sequence[Partition], tp.Dict[str, Partition]]
    ) -> ApplyCaller:
        if isinstance(partitions, dict):
            partitions = tuple(partitions.values())
        module: M = self.merge(partitions)

        def _context(fn, *args, **kwargs):
            out = fn(*args, **kwargs)
            partitions, _ = module.partition()
            return out, partitions

        return CallableProxy(_context, module)  # type: ignore


class Bounded(tp.NamedTuple, tp.Generic[M]):
    partitions: tp.Dict[str, Partition]
    moduledef: ModuleDef[M]

    @property
    def module(self) -> M:
        def _context(apply, *args, **kwargs):
            out, updates = apply(*args, **kwargs)
            self.partitions.update(updates)
            return out

        return CallableProxy(_context, self.moduledef.apply)  # type: ignore
