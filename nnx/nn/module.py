import dataclasses
import jax
from simple_pytree import Pytree
from nnx import partitioning
from nnx import scope_lib
import refx
import typing as tp
import jax.tree_util as jtu

M = tp.TypeVar("M", bound="Module")


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
    def deref(self: M) -> tp.Tuple[M, refx.DagDef]:
        return refx.deref(self)

    def reref(self: M, dagdef: refx.DagDef) -> M:
        return refx.reref(self, dagdef)

    def clone(self: M) -> M:
        return refx.clone(self)

    def make_rng(self, collection: str) -> jax.random.KeyArray:
        return scope_lib.make_rng(collection)

    def get_flag(self, name: str, default: tp.Any = dataclasses.MISSING) -> tp.Any:
        return scope_lib.get_flag(name, default)

    def __getitem__(self, filter: str) -> refx.Partition:
        return partitioning.get_partition(self.deref()[0], filter)

    def __setitem__(self, filter: str, value: refx.Partition):
        refx.update_refs(partitioning.get_partition(self, filter), value)

    def collections(self) -> tp.FrozenSet[str]:
        return frozenset(
            x.collection
            for x in jtu.tree_leaves(
                self, is_leaf=lambda x: isinstance(x, refx.Referential)
            )
            if isinstance(x, refx.Referential)
            if isinstance(x.collection, str)
        )

    @tp.overload
    def partition(
        self: M,
        *,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Tuple[tp.Dict[str, refx.Partition], "ModuleDef[M]"]:
        ...

    @tp.overload
    def partition(
        self: M,
        collection: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Tuple[refx.Partition, "ModuleDef[M]",]:
        ...

    @tp.overload
    def partition(
        self: M,
        collection: str,
        *extract_colelctions: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Tuple[tp.Tuple[refx.Partition, ...], "ModuleDef[M]",]:
        ...

    def partition(
        self: M,
        *extract_collections: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Tuple[
        tp.Union[
            tp.Dict[str, refx.Partition], tp.Tuple[refx.Partition, ...], refx.Partition
        ],
        "ModuleDef[M]",
    ]:
        collections = list(self.collections())
        module, dagdef = self.deref()
        partitions, treedef = partitioning.tree_partition(
            module, *collections, is_leaf=is_leaf
        )
        moduledef = ModuleDef(dagdef, treedef)

        if all(x is refx.NOTHING for x in partitions[-1].values()):
            partitions = partitions[:-1]
        else:
            # add "rest" as a reserved name for the rest of the tree
            collections.append("rest")

        partitions = dict(zip(collections, partitions))

        if extract_collections:
            if set(extract_collections) != set(collections):
                raise ValueError(
                    f"extract_colelctions contain all collections: "
                    f"{extract_collections} != {collections}"
                )

            if len(extract_collections) == 1:
                partitions = partitions[extract_collections[0]]
            else:
                partitions = tuple(partitions[x] for x in extract_collections)

        return partitions, moduledef

    @classmethod
    def init(
        cls: tp.Type[M],
        rngs: tp.Union[
            tp.Dict[str, jax.random.KeyArray], jax.random.KeyArray, None
        ] = None,
        **flags: tp.Hashable,
    ) -> tp.Type[M]:
        if rngs is None:
            rngs = {}
        elif isinstance(rngs, jax.Array):
            rngs = {"params": rngs}

        scope = scope_lib.Scope.from_keys_and_flags(rngs, flags)

        def _context(fn, *args, **kwargs):
            with scope_lib.scope(scope):
                return fn(*args, **kwargs)

        return CallableProxy(_context, cls)  # type: ignore

    def apply(
        self: M,
        *,
        rngs: tp.Optional[tp.Dict[str, jax.random.KeyArray]] = None,
        **flags: tp.Hashable,
    ) -> M:
        if rngs is None:
            rngs = {}

        scope = scope_lib.Scope.from_keys_and_flags(rngs, flags)

        def _context(fn, *args, **kwargs):
            with scope_lib.scope(scope):
                return fn(*args, **kwargs)

        return CallableProxy(_context, self)  # type: ignore


class ApplyCaller(tp.Protocol):
    def __getattr__(self, __name) -> "ApplyCaller":
        ...

    def __call__(
        self, *args, **kwargs
    ) -> tp.Tuple[tp.Any, tp.Dict[str, refx.Partition]]:
        ...


class ModuleDefValue(tp.NamedTuple):
    reref_dagdef: refx.DagDef
    partition_treedef: jtu.PyTreeDef


class ModuleDef(tp.Generic[M]):
    __slots__ = ("_reref_dagdef", "_partition_treedef")

    def __init__(self, reref_dagdef: refx.DagDef, partition_treedef: jtu.PyTreeDef):
        self._reref_dagdef = reref_dagdef
        self._partition_treedef = partition_treedef

    @property
    def reref_dagdef(self) -> refx.DagDef:
        return self._reref_dagdef

    @property
    def partition_treedef(self) -> jtu.PyTreeDef:
        return self._partition_treedef

    def apply(
        self,
        partitions: tp.Union[tp.Sequence[refx.Partition], tp.Dict[str, refx.Partition]],
        *,
        rngs: tp.Optional[tp.Dict[str, jax.random.KeyArray]] = None,
        flags: tp.Optional[tp.Dict[str, tp.Hashable]] = None,
    ) -> ApplyCaller:
        if flags is None:
            flags = {}
        if rngs is None:
            rngs = {}
        if isinstance(partitions, dict):
            partitions = tuple(partitions.values())
        module: M = self.merge(partitions)
        scope = scope_lib.Scope.from_keys_and_flags(rngs, flags)

        def _context(fn, *args, **kwargs):
            with scope_lib.scope(scope):
                out = fn(*args, **kwargs)
                partitions, _ = module.partition()
                return out, partitions

        return CallableProxy(_context, module)  # type: ignore

    def merge(
        self,
        partitions: tp.Union[tp.Sequence[refx.Partition], tp.Dict[str, refx.Partition]],
    ) -> M:
        if isinstance(partitions, dict):
            partitions = tuple(partitions.values())
        module: M = partitioning.merge_partitions(partitions, self.partition_treedef)
        return module.reref(self.reref_dagdef)


def _moduledef_flatten(
    x: ModuleDef[M],
) -> tp.Tuple[tp.Tuple[()], tp.Tuple[refx.DagDef, jtu.PyTreeDef]]:
    return (), (x.reref_dagdef, x.partition_treedef)


def _moduledef_unflatten(
    aux_data: tp.Tuple[refx.DagDef, jtu.PyTreeDef], _: tp.Tuple[()]
) -> ModuleDef[M]:
    return ModuleDef(*aux_data)


jtu.register_pytree_node(ModuleDef, _moduledef_flatten, _moduledef_unflatten)
