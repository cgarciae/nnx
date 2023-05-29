from abc import ABC
import dataclasses
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
from nnx.reference import Deref, Index, Partition, Ref, Referential, Value
import typing as tp
import jax.tree_util as jtu
import builtins
from nnx import partitioning, tracers

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="Module")
P = tp.TypeVar(
    "P",
    bound=tp.Union[Partition, tp.Tuple[Partition, ...], tp.Dict[str, Partition]],
)
Path = tp.Tuple[str, ...]
State = tp.Dict[Path, tp.Any]
StateLike = tp.Mapping[Path, tp.Any]


class ApplyCaller(tp.Protocol):
    def __getattr__(self, __name) -> "ApplyCaller":
        ...

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, tp.Dict[str, Partition]]:
        ...


class ModuleDef(tp.Generic[M]):
    __slots__ = ("_type", "_index", "_submodules", "_static_fields")

    def __init__(
        self,
        type: tp.Type[M],
        index: int,
        submodules: tp.Tuple[tp.Tuple[str, tp.Union["ModuleDef[Module]", int]], ...],
        static_fields: tp.Tuple[tp.Tuple[str, tp.Any], ...],
    ):
        self._type = type
        self._index = index
        self._submodules = submodules
        self._static_fields = static_fields

    def __hash__(self) -> int:
        return hash((self._type, self._submodules, self._static_fields))

    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, ModuleDef):
            return False
        return (
            self._type == other._type
            and self._submodules == other._submodules
            and self._static_fields == other._static_fields
        )

    @property
    def type(self) -> tp.Type[M]:
        return self._type

    @property
    def index(self) -> int:
        return self._index

    @property
    def submodules(
        self,
    ) -> tp.Tuple[tp.Tuple[str, tp.Union["ModuleDef[Module]", int]], ...]:
        return self._submodules

    @property
    def static_fields(self) -> tp.Tuple[tp.Tuple[str, tp.Any], ...]:
        return self._static_fields

    def reref(
        self,
        partitions: tp.Union[
            Partition, tp.Tuple[Partition, ...], tp.Dict[str, Partition]
        ],
    ) -> M:
        if not isinstance(partitions, (Partition, tuple, dict)):
            raise TypeError(
                f"partitions must be a Partition, tuple of Partitions, or dict of Partitions, "
                f"got {type(partitions).__name__}"
            )
        if isinstance(partitions, Partition):
            partition = partitions
        elif isinstance(partitions, tuple):
            partition = _merge_state(partitions)
        else:
            partition = _merge_state(partitions.values())

        return _reref(partition, self)

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


def _moddef_flatten(moddef: ModuleDef[M]):
    return (), (moddef._type, moddef._index, moddef._submodules, moddef._static_fields)


def _moddef_unflatten(
    metadata: tp.Tuple[
        tp.Type[M],
        int,
        tp.Tuple[tp.Tuple[str, tp.Union["ModuleDef[Module]", int]], ...],
        tp.Tuple[tp.Tuple[str, tp.Any], ...],
    ],
    _,
) -> ModuleDef[M]:
    return ModuleDef(*metadata)


jtu.register_pytree_node(ModuleDef, _moddef_flatten, _moddef_unflatten)


class DerefedMod(tp.Tuple[P, ModuleDef[M]]):
    @property
    def partitions(self) -> P:
        return self[0]

    @property
    def moduledef(self) -> ModuleDef[M]:
        return self[1]

    def reref(self) -> M:
        return self.moduledef.reref(self.partitions)

    @property
    def apply(self) -> ApplyCaller:
        return self.moduledef.apply(self.partitions)


def _derefedmod_flatten(bounded: DerefedMod[P, M]):
    return tuple(bounded), None


def _derefedmod_unflatten(_, values):
    return DerefedMod(values)


jtu.register_pytree_node(DerefedMod, _derefedmod_flatten, _derefedmod_unflatten)


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


class Module(ABC):
    if not tp.TYPE_CHECKING:

        def __getattribute__(self, __name: str) -> Any:
            value = object.__getattribute__(self, __name)
            if isinstance(value, Ref):
                return value.value
            return value

    def get_ref(self, __name: str) -> Ref[tp.Any]:
        if __name not in vars(self):
            raise AttributeError(f"Module has no attribute {__name}")

        return vars(self)[__name]

    @property
    def refs(self) -> tp.Mapping[str, Ref[tp.Any]]:
        return MappingProxyType(
            {k: v for k, v in vars(self).items() if isinstance(v, Ref)}
        )

    def __hash__(self) -> int:
        return id(self)

    def deref(self: M) -> DerefedMod[Partition, M]:
        state, moduledef = _deref(self)
        partition = Partition(state)
        return DerefedMod((partition, moduledef))

    def clone(self: M) -> M:
        return self.deref().reref()

    def collections(self) -> tp.Set[str]:
        return {
            x.collection
            for x in jtu.tree_leaves(self, is_leaf=lambda x: isinstance(x, Referential))
            if isinstance(x, Referential)
        }

    def partition_general(
        self: M, *filters: partitioning.CollectionFilter
    ) -> DerefedMod[tp.Tuple[Partition, ...], M]:
        partition, moduledef = self.deref()
        partitions = _split_partition(partition, *filters)
        return DerefedMod((partitions, moduledef))

    @tp.overload
    def partition(
        self: M,
        collection: None = None,
        second: None = None,
        /,
    ) -> DerefedMod[tp.Dict[str, Partition], M]:
        ...

    @tp.overload
    def partition(
        self: M, collection: str, second: None = None, /
    ) -> DerefedMod[Partition, M]:
        ...

    @tp.overload
    def partition(
        self: M, collection: str, second: str, /, *collections: str
    ) -> DerefedMod[tp.Tuple[Partition, ...], M]:
        ...

    def partition(
        self: M,
        collection: tp.Optional[str] = None,
        second: tp.Optional[str] = None,
        /,
        *collections: str,
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
            partitions, moddef = _partition_by_collection(self)
        elif len(collections) == 1:
            partitions, moddef = _partition_by_collection(self, collections[0])
        else:
            partitions, moddef = _partition_by_collection(
                self, collections[0], collections[1], *collections[2:]
            )

        return DerefedMod((partitions, moddef))  # type: ignore

    @tp.overload
    def get_partition(
        self,
        filter: partitioning.CollectionFilter,
        /,
    ) -> Partition:
        ...

    @tp.overload
    def get_partition(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[Partition, ...]:
        ...

    def get_partition(
        self, *filters: partitioning.CollectionFilter
    ) -> tp.Union[Partition, tp.Tuple[Partition, ...]]:
        if len(filters) == 0:
            raise ValueError("Expected at least one filter")

        (*partitions, _rest), _ = self.partition_general(*filters)

        assert len(partitions) == len(filters)

        if len(partitions) == 1:
            partitions = partitions[0]
        else:
            partitions = tuple(partitions)

        return partitions

    def update(
        self,
        partitions: tp.Union[
            Partition, tp.Tuple[Partition, ...], tp.Dict[str, Partition]
        ],
    ) -> None:
        if isinstance(partitions, Partition):
            new_state = partitions
        else:
            if isinstance(partitions, dict):
                partitions = tuple(partitions.values())

            new_state = _merge_partitions(partitions)

        # sort by Values first, then by other values
        new_state = dict(
            sorted(new_state.items(), key=lambda x: 1 if isinstance(x[1], Value) else 2)
        )

        current_state = self.ref_dict()
        context_trace = tracers.get_top_trace(
            [x.value if isinstance(x, Ref) else x for x in current_state.values()]
        )

        for path, new_value in new_state.items():
            if isinstance(new_value, Value):
                if path in current_state:
                    assert isinstance(current_state[path], Ref)
                    current_state[path].value = new_value.value
                else:
                    current_state[path] = new_value.to_ref(context_trace)
                    _set_value_at_path(self, path, current_state[path])
            elif isinstance(new_value, Index):
                if path not in current_state:
                    if new_value.val_path not in current_state:
                        raise ValueError(
                            f"No Value for Index at path {path}, "
                            f"expected Value at {new_value.val_path}"
                        )

                    _set_value_at_path(self, path, current_state[new_value.val_path])
            else:
                _set_value_at_path(self, path, new_value)

    @tp.overload
    def ref_dict(self) -> tp.Dict[Path, tp.Any]:
        ...

    @tp.overload
    def ref_dict(self, sep: str) -> tp.Dict[str, tp.Any]:
        ...

    def ref_dict(
        self, sep: tp.Optional[str] = None
    ) -> tp.Union[tp.Dict[Path, tp.Any], tp.Dict[str, tp.Any]]:
        state = _ref_dict(self)

        if sep is not None:
            state = {sep.join(path): leaf for path, leaf in state.items()}
        return state


def _deref(module: M) -> tp.Tuple[State, ModuleDef[M]]:
    module_index: tp.Dict[Module, int] = {}
    ref_path: tp.Dict[Ref[tp.Any], Path] = {}
    path: Path = ()
    state: tp.Dict[Path, tp.Any] = {}

    moduledef = _deref_recursive(module, module_index, ref_path, path, state)
    assert isinstance(moduledef, ModuleDef)

    return state, moduledef


def _deref_recursive(
    module: M,
    module_index: tp.Dict[Module, int],
    ref_path: tp.Dict[Ref[tp.Any], Path],
    path: Path,
    state: tp.Dict[Path, tp.Any],
) -> tp.Union[ModuleDef[M], int]:
    if module in module_index:
        return module_index[module]

    submodules = []
    static_fields = []

    for name, value in vars(module).items():
        value_path = (*path, name)
        if isinstance(value, Module):
            submodule_dag = _deref_recursive(
                value, module_index, ref_path, value_path, state
            )
            submodules.append((name, submodule_dag))
        elif isinstance(value, Ref):
            if value not in ref_path:
                ref_path[value] = value_path
                value = value.to_value()
            else:
                value = value.to_index(ref_path[value])

            state[value_path] = value
        elif isinstance(value, (jax.Array, np.ndarray)):
            state[value_path] = value
        else:
            static_fields.append((name, value))

    index = len(module_index)
    module_dag = ModuleDef(
        type=type(module),
        index=index,
        submodules=tuple(submodules),
        static_fields=tuple(static_fields),
    )
    module_index[module] = index
    return module_dag


def _ref_dict(module: Module) -> State:
    seen_modules: tp.Set[Module] = set()
    path: Path = ()
    state: State = {}

    _ref_dict_recursive(module, seen_modules, path, state)
    return state


def _ref_dict_recursive(
    module: Module,
    seen_modules: tp.Set[Module],
    path: Path,
    state: tp.Dict[Path, tp.Any],
) -> None:
    if module in seen_modules:
        return

    for name, value in vars(module).items():
        value_path = (*path, name)
        if isinstance(value, Module):
            _ref_dict_recursive(value, seen_modules, value_path, state)
        elif isinstance(value, Ref):
            state[value_path] = value
        elif isinstance(value, (jax.Array, np.ndarray)):
            state[value_path] = value


def _reref(state: StateLike, moduledef: ModuleDef[M]) -> M:
    index_module: tp.Dict[int, Module] = {}
    module = _build_module(moduledef, index_module)
    state = _reref_state(state)

    for path, value in state.items():
        _set_value_at_path(module, path, value)

    return module


def _set_value_at_path(module: M, path: Path, value: tp.Any) -> M:
    if len(path) == 1:
        setattr(module, path[0], value)
    else:
        _set_value_at_path(getattr(module, path[0]), path[1:], value)


def _reref_state(state: StateLike) -> State:
    new_state: State = {}

    context_trace = tracers.get_top_trace(state)

    for path, value in state.items():
        if path in new_state:
            continue
        elif isinstance(value, Value):
            new_state[path] = value.to_ref(context_trace)
        elif isinstance(value, Index):
            if value.val_path in new_state:
                assert isinstance(new_state[value.val_path], Ref)
                new_state[path] = new_state[value.val_path]
            else:
                # if we visite an index before its value, we need to
                # create the ref and add both paths to the new state
                deref_value = state[value.val_path]
                assert isinstance(deref_value, Value)
                ref = deref_value.to_ref(context_trace)
                new_state[value.val_path] = ref
                new_state[path] = ref
        else:
            new_state[path] = value

    return new_state


def _build_module(
    moduledef: tp.Union[ModuleDef[M], int],
    index_module: tp.Dict[int, Module],
) -> M:
    if isinstance(moduledef, int):
        return index_module[moduledef]  # type: ignore

    assert moduledef.index not in index_module

    submodules = {
        name: _build_module(submodule, index_module)
        for name, submodule in moduledef.submodules
    }

    module = object.__new__(moduledef.type)
    vars(module).update(moduledef.static_fields)
    vars(module).update(submodules)

    return module


def _merge_state(partitions: tp.Iterable[StateLike]) -> State:
    new_state: State = {}

    for state in partitions:
        new_state.update(state)

    return new_state


@tp.overload
def _partition_by_collection(module: M) -> DerefedMod[tp.Dict[str, Partition], M]:
    ...


@tp.overload
def _partition_by_collection(module: M, collection: str) -> DerefedMod[Partition, M]:
    ...


@tp.overload
def _partition_by_collection(
    module: M, collection: str, second: str, *rest: str
) -> DerefedMod[tp.Tuple[Partition, ...], M]:
    ...


def _partition_by_collection(
    module: M, *collections: str
) -> tp.Union[
    DerefedMod[Partition, M],
    DerefedMod[tp.Tuple[Partition, ...], M],
    DerefedMod[tp.Dict[str, Partition], M],
]:
    partition, moddef = module.deref()

    num_collections = len(collections)

    if num_collections == 0:
        collections = tuple(
            set(x.collection for x in partition.values() if isinstance(x, Deref))
        )
        if "rest" in collections:
            raise ValueError("Found reserved 'rest' collection name in module Refs")

    partitions = _split_partition(partition, *collections)

    if len(partitions[-1]) == 0:
        partitions = partitions[:-1]
    else:
        if num_collections == 0:
            collections = (*collections, "rest")
        elif "rest" not in collections:
            raise ValueError(
                f"Expected 'rest' collection to be in collections: {collections}"
            )

    if len(collections) != len(partitions):
        raise ValueError(
            f"Expected the number of collections ({len(collections)}) "
            f"to be equal to the number of partitions ({len(partitions)})."
        )

    if num_collections == 0:
        partitions = dict(zip(collections, partitions))
    elif num_collections == 1:
        partitions = partitions[0]

    return DerefedMod((partitions, moddef))  # type: ignore


def _split_partition(
    partition: Partition,
    *filters: partitioning.CollectionFilter,
) -> tp.Tuple[Partition, ...]:
    predicates = tuple(map(partitioning.to_predicate, filters))

    # we have n + 1 partitions, where n is the number of predicates
    # the last partition is for values that don't match any predicate
    partitions: tp.Tuple[State, ...] = tuple({} for _ in range(len(predicates) + 1))

    for path, value in partition.items():
        for i, predicate in enumerate(predicates):
            if predicate(path, value):
                partitions[i][path] = value
                break
        else:
            # if we didn't break, set leaf to last partition
            partitions[-1][path] = value

    return tuple(Partition(x) for x in partitions)


def _merge_partitions(
    partitions: tp.Iterable[Partition],
) -> Partition:
    state: State = {}

    for partition in partitions:
        state.update(partition)

    return Partition(state)
