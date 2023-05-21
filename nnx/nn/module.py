import dataclasses
import jax
from simple_pytree import Pytree
from nnx import partitioning
from nnx import scope_lib
from nnx.reference import (
    DagDef,
    Partition,
    Referential,
    clone,
    deref,
    reref,
    update_refs,
)
import typing as tp
import jax.tree_util as jtu
import builtins

from nnx.rng_stream import RngStream

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

    def make_rng(self, collection: str) -> jax.random.KeyArray:
        return scope_lib.make_rng(collection)

    def get_flag(self, name: str, default: tp.Any = dataclasses.MISSING) -> tp.Any:
        return scope_lib.get_flag(name, default)

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
    ) -> tp.Tuple[tp.Dict[str, Partition], "ModuleDef[M]"]:
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
    ) -> tp.Tuple[
        tp.Union[tp.Dict[str, Partition], tp.Tuple[Partition, ...], Partition],
        "ModuleDef[M]",
    ]:
        partitions, dagdef = partitioning.collection_partition(
            self, *collections, is_leaf=is_leaf
        )
        moduledef = ModuleDef(dagdef.indexes, dagdef.treedef)

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
        rngs: tp.Optional[tp.Mapping[str, jax.random.KeyArray]] = None,
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

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, tp.Dict[str, Partition]]:
        ...


class ModuleDef(DagDef[M]):
    def apply(
        self,
        partitions: tp.Union[tp.Sequence[Partition], tp.Dict[str, Partition]],
        *,
        rngs: tp.Optional[tp.Mapping[str, tp.Union[jax.Array, RngStream]]] = None,
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
