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
    def deref(self: M) -> M:
        return refx.deref(self)

    def reref(self: M) -> M:
        return refx.reref(self)

    def make_rng(self, collection: str) -> jax.random.KeyArray:
        return scope_lib.make_rng(collection)

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
        *extract_colelctions: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Tuple[tp.Tuple[refx.Partition, ...], "ModuleDef[M]",]:
        ...

    def partition(
        self: M,
        *extract_collections: str,
        is_leaf: tp.Optional[partitioning.LeafPredicate] = None,
    ) -> tp.Tuple[
        tp.Union[tp.Dict[str, refx.Partition], tp.Tuple[refx.Partition, ...]],
        "ModuleDef[M]",
    ]:
        collections = list(self.collections())
        partitions, treedef = partitioning.tree_partition(
            self.deref(), *collections, is_leaf=is_leaf
        )
        moduledef = ModuleDef(treedef)

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

            partitions = tuple(partitions[x] for x in extract_collections)

        return partitions, moduledef

    @classmethod
    def init(
        cls: tp.Type[M],
        rngs: tp.Union[
            tp.Dict[str, jax.random.KeyArray], jax.random.KeyArray, None
        ] = None,
        *,
        flags: tp.Optional[tp.Dict[str, tp.Hashable]] = None,
    ) -> tp.Type[M]:
        if flags is None:
            flags = {}
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
        flags: tp.Optional[tp.Dict[str, tp.Hashable]] = None,
    ) -> M:
        if flags is None:
            flags = {}
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


class ModuleDef(refx.Static[jtu.PyTreeDef], tp.Generic[M]):
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
                partitions, _ = module.deref().partition()
                return out, partitions

        return CallableProxy(_context, module)  # type: ignore

    def merge(
        self,
        partitions: tp.Union[tp.Sequence[refx.Partition], tp.Dict[str, refx.Partition]],
    ) -> M:
        if isinstance(partitions, dict):
            partitions = tuple(partitions.values())
        module: M = partitioning.merge_partitions(partitions, self.value)
        return module.reref()


def _moduledef_flatten(x: ModuleDef[M]) -> tp.Tuple[tp.Tuple[()], jtu.PyTreeDef]:
    return (), x.value


def _moduledef_unflatten(aux_data: jtu.PyTreeDef, _: tp.Tuple[()]) -> ModuleDef[M]:
    return ModuleDef(aux_data)


jtu.register_pytree_node(ModuleDef, _moduledef_flatten, _moduledef_unflatten)