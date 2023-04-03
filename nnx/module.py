import dataclasses
import jax
from simple_pytree import Pytree
from nnx import partitioning
from nnx import scope_lib
import refx
import typing as tp
import jax.tree_util as jtu

M = tp.TypeVar("M", bound="Module")


@dataclasses.dataclass
class ScopeContextCaller:
    _scope_context_scope: scope_lib.Scope
    _scope_context_obj: tp.Any

    def __call__(self, *args, **kwargs):
        with scope_lib.scope(self._scope_context_scope):
            return self._scope_context_obj(*args, **kwargs)

    def __getattr__(self, name) -> "ScopeContextCaller":
        return ScopeContextCaller(
            self._scope_context_scope, getattr(self._scope_context_obj, name)
        )


class Module(Pytree):
    def deref(self: M) -> M:
        return refx.deref(self)

    def reref(self: M) -> M:
        return refx.reref(self)

    def __getitem__(self, filter: str) -> refx.Partition:
        return partitioning.get_partition(self.deref(), filter)

    def __setitem__(self, filter: str, value: refx.Partition):
        refx.update_from(partitioning.get_partition(self, filter), value)

    def collections(self) -> tp.FrozenSet[str]:
        return frozenset(
            x.collection
            for x in jtu.tree_leaves(
                self, is_leaf=lambda x: isinstance(x, refx.Referential)
            )
            if isinstance(x, refx.Referential)
            if isinstance(x.collection, str)
        )

    def partition(
        self: M, is_leaf: tp.Optional[partitioning.LeafPredicate] = None
    ) -> tp.Tuple[tp.Dict[str, refx.Partition], "ModuleDef[M]"]:
        collections = list(self.collections())
        partitions, treedef = partitioning.tree_partition(
            self.deref(), *collections, is_leaf=is_leaf
        )
        moduledef = ModuleDef(treedef)
        # add "rest" as a unexistant collection name reserved for the rest of the tree
        # tree_partition will always return a partition for "rest"
        collections.append("rest")

        return dict(zip(collections, partitions)), moduledef

    @classmethod
    def init(
        cls: tp.Type[M],
        *,
        rngs: tp.Optional[tp.Dict[str, jax.random.KeyArray]] = None,
        flags: tp.Optional[tp.Dict[str, tp.Hashable]] = None,
    ) -> tp.Type[M]:
        if flags is None:
            flags = {}
        if rngs is None:
            rngs = {}

        scope = scope_lib.Scope.from_keys_and_flags(rngs, flags)
        return ScopeContextCaller(scope, cls)  # type: ignore

    def apply(
        self: M,
        *,
        rngs: tp.Optional[tp.Dict[str, jax.random.KeyArray]] = None,
        flags: tp.Optional[tp.Dict[str, tp.Hashable]] = None,
    ) -> M:
        if flags is None:
            flags = {}
        if rngs is None:
            rngs = {}

        scope = scope_lib.Scope.from_keys_and_flags(rngs, flags)
        return ScopeContextCaller(scope, self)  # type: ignore


class ModuleDef(refx.Static[jtu.PyTreeDef], tp.Generic[M]):
    def apply(
        self,
        partitions: tp.Dict[str, refx.Partition],
        *,
        rngs: tp.Optional[tp.Dict[str, jax.random.KeyArray]] = None,
        flags: tp.Optional[tp.Dict[str, tp.Hashable]] = None,
    ) -> M:
        module: M = partitioning.merge_partitions(
            tuple(partitions.values()), self.value
        )
        return module.apply(rngs=rngs, flags=flags)
