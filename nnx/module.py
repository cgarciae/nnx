import dataclasses
from nnx.pytree import Pytree
from nnx import partitioning
from nnx.reference import (
    NOTHING,
    P,
    DagDef,
    Derefed,
    Partition,
    Referential,
    clone,
    deref,
    reref,
    update_refs,
    _to_str_path,
    _merge_partitions,
)
import typing as tp
import jax.tree_util as jtu
import builtins


A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="Module")


class ApplyCaller(tp.Protocol):
    def __getattr__(self, __name) -> "ApplyCaller":
        ...

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, tp.Dict[str, Partition]]:
        ...


class ModuleDef(DagDef[M]):
    def apply(
        self,
        partitions: tp.Union[
            Partition, tp.Tuple[Partition, ...], tp.Dict[str, Partition]
        ],
    ) -> ApplyCaller:
        module: M = self.reref(partitions)

        def _context(fn, *args, **kwargs) -> tp.Tuple[tp.Any, tp.Dict[str, Partition]]:
            out = fn(*args, **kwargs)
            updates, _ = module.partition()
            return out, updates

        return CallableProxy(_context, module)  # type: ignore


class DerefedMod(tp.Tuple[P, ModuleDef[M]]):
    @property
    def partitions(self) -> P:
        return self[0]

    @property
    def moduledef(self) -> ModuleDef[M]:
        return self[1]

    def reref(self) -> M:
        return reref(self.partitions, self.moduledef)

    @property
    def apply(self) -> ApplyCaller[P]:
        return self.moduledef.apply(self.partitions)


def _flatten_bounded(bounded: DerefedMod[P, M]):
    return tuple(bounded), None


def _unflatten_bounded(_, values):
    return DerefedMod(values)


jtu.register_pytree_node(DerefedMod, _flatten_bounded, _unflatten_bounded)


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
    def deref(self: M) -> DerefedMod[Partition, M]:
        partition, dagdef = deref(self)
        return DerefedMod((partition, ModuleDef.from_value(dagdef)))

    def clone(self: M) -> M:
        return clone(self)

    @tp.overload
    def __getitem__(self, collections: str) -> Partition:
        ...

    @tp.overload
    def __getitem__(self, collections: tp.Tuple[str, ...]) -> tp.Tuple[Partition, ...]:
        ...

    @tp.overload
    def __getitem__(
        self, collections: tp.Union[str, tp.Tuple[str, ...]]
    ) -> tp.Union[Partition, tp.Tuple[Partition, ...]]:
        ...

    def __getitem__(
        self, collections: tp.Union[str, tp.Tuple[str, ...]]
    ) -> tp.Union[Partition, tp.Tuple[Partition, ...]]:
        if len(collections) < 1:
            raise ValueError("Must specify at least one collection")

        if isinstance(collections, str):
            collections = (collections,)

        return partitioning.get_partition(self, *collections)

    def __setitem__(
        self,
        _: SetItemType,
        value: tp.Union[Partition, tp.Tuple[Partition, ...], tp.Dict[str, Partition]],
    ):
        update_refs(self, value)

    def collections(self) -> tp.Set[str]:
        return {
            x.collection
            for x in jtu.tree_leaves(self, is_leaf=lambda x: isinstance(x, Referential))
            if isinstance(x, Referential)
        }

    @tp.overload
    def partition(
        self: M,
        collection: None = None,
        second: None = None,
        /,
        *,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> DerefedMod[tp.Dict[str, Partition], M]:
        ...

    @tp.overload
    def partition(
        self: M,
        collection: str,
        second: None = None,
        /,
        *,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> DerefedMod[Partition, M]:
        ...

    @tp.overload
    def partition(
        self: M,
        collection: str,
        second: str,
        /,
        *collections: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> DerefedMod[tp.Tuple[Partition, ...], M]:
        ...

    def partition(
        self: M,
        collection: tp.Optional[str] = None,
        second: tp.Optional[str] = None,
        /,
        *collections: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Union[
        DerefedMod[Partition, M],
        DerefedMod[tp.Tuple[Partition, ...], M],
        DerefedMod[tp.Dict[str, Partition], M],
    ]:
        if second is not None:
            collections = (second, *collections)

        if collection is not None:
            collections = (collection, *collections)

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
                self, collections[0], collections[1], *collections[2:], is_leaf=is_leaf
            )

        moduledef = ModuleDef.from_value(dagdef)

        return DerefedMod((partitions, moduledef))

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
