from abc import ABC
import dataclasses
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
from zmq import has
from nnx.reference import Deref, Index, State, Ref, Referential, Value
import typing as tp
import jax.tree_util as jtu
import builtins
from nnx import partitioning, tracers

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="Module")
P = tp.TypeVar(
    "P",
    bound=tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
)
Path = tp.Tuple[str, ...]
StateDict = tp.Dict[Path, tp.Any]
StateMapping = tp.Mapping[Path, tp.Any]


class ApplyCaller(tp.Protocol):
    def __getattr__(self, __name) -> "ApplyCaller":
        ...

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, tp.Dict[str, State]]:
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

    def __repr__(self) -> str:
        return (
            f"ModuleDef(type={self._type.__name__}, index={self._index}, "
            f"submodules={self._submodules}, static_fields={self._static_fields})"
        )

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
        states: tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> M:
        if not isinstance(states, (State, tuple, dict)):
            raise TypeError(
                f"states must be a State, tuple of States, or dict of States, "
                f"got {type(states).__name__}"
            )
        if isinstance(states, State):
            state = states
        elif isinstance(states, tuple):
            state = _merge_state(states)
        else:
            state = _merge_state(states.values())

        return _reref(state, self)

    def apply(
        self,
        states: tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> ApplyCaller:
        module: M = self.reref(states)

        def _context(fn, *args, **kwargs) -> tp.Tuple[tp.Any, tp.Dict[str, State]]:
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


class StateDef(tp.Tuple[P, ModuleDef[M]]):
    @property
    def states(self) -> P:
        return self[0]

    @property
    def moduledef(self) -> ModuleDef[M]:
        return self[1]

    def reref(self) -> M:
        return self.moduledef.reref(self.states)

    @property
    def apply(self) -> ApplyCaller:
        return self.moduledef.apply(self.states)


def _derefedmod_flatten(bounded: StateDef[P, M]):
    return tuple(bounded), None


def _derefedmod_unflatten(_, values):
    return StateDef(values)


jtu.register_pytree_node(StateDef, _derefedmod_flatten, _derefedmod_unflatten)


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

        def __setattr__(self, __name: str, value: Any) -> None:
            vars_dict = vars(self)
            if (
                __name in vars_dict
                and isinstance(vars_dict[__name], Ref)
                and not isinstance(value, Ref)
            ):
                raise TypeError(
                    f"Trying to set a Ref attribute '{__name}' to a non-Ref value of "
                    f"type '{type(value).__name__}'. To update a Ref attribute "
                    f"use 'del module.{__name}' first to remove the attribute and then "
                    f"set the attribute to the new value."
                )

            object.__setattr__(self, __name, value)

    def __hash__(self) -> int:
        return id(self)

    def deref(self: M) -> StateDef[State, M]:
        state, moduledef = _deref(self)
        state = State(state)
        return StateDef((state, moduledef))

    def clone(self: M) -> M:
        return self.deref().reref()

    def collections(self) -> tp.Set[str]:
        return {
            x.collection
            for x in jtu.tree_leaves(self, is_leaf=lambda x: isinstance(x, Referential))
            if isinstance(x, Referential)
        }

    @tp.overload
    def partition(
        self: M,
        first: None = None,
        second: None = None,
        /,
    ) -> StateDef[tp.Dict[str, State], M]:
        ...

    @tp.overload
    def partition(
        self: M, first: partitioning.CollectionFilter, second: None = None, /
    ) -> StateDef[State, M]:
        ...

    @tp.overload
    def partition(
        self: M,
        first: partitioning.CollectionFilter,
        second: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> StateDef[tp.Tuple[State, ...], M]:
        ...

    def partition(
        self: M,
        first: tp.Optional[partitioning.CollectionFilter] = None,
        second: tp.Optional[partitioning.CollectionFilter] = None,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Union[
        StateDef[State, M],
        StateDef[tp.Tuple[State, ...], M],
        StateDef[tp.Dict[str, State], M],
    ]:
        if second is not None:
            filters = (second, *filters)

        if first is not None:
            filters = (first, *filters)

        if len(filters) == 0:
            states, moddef = _partition_by_collection(self)
        else:
            state, moddef = self.deref()
            (*states, rest) = _split_state(state, *filters)

            if len(rest) > 0:
                raise ValueError(
                    f"Non-exhaustive filters, got a non-empty remainder: {rest}.\n"
                    f"Use `...` to match all remaining elements."
                )

            states = tuple(states)

        return StateDef((states, moddef))  # type: ignore

    @tp.overload
    def get(
        self,
        filter: partitioning.CollectionFilter,
        /,
    ) -> State:
        ...

    @tp.overload
    def get(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[State, ...]:
        ...

    def get(
        self, *filters: partitioning.CollectionFilter
    ) -> tp.Union[State, tp.Tuple[State, ...]]:
        if len(filters) == 0:
            raise ValueError("Expected at least one filter")

        state, _ = self.deref()
        (*states, _rest) = _split_state(state, *filters)

        assert len(states) == len(filters)

        if len(states) == 1:
            return states[0]
        else:
            return tuple(states)

    @property
    def update(
        self,
    ) -> tp.Callable[
        [tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]]], None
    ]:
        return lambda states: self._update(states)

    @update.setter
    def update(
        self,
        value: tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> None:
        self._update(value)

    def _update(
        self,
        states: tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> None:
        if isinstance(states, State):
            new_state = states
        else:
            if isinstance(states, dict):
                states = tuple(states.values())

            new_state = _merge_partitions(states)

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


def _deref(module: M) -> tp.Tuple[StateDict, ModuleDef[M]]:
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


def _ref_dict(module: Module) -> StateDict:
    seen_modules: tp.Set[Module] = set()
    path: Path = ()
    state: StateDict = {}

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


def _reref(state: StateMapping, moduledef: ModuleDef[M]) -> M:
    index_module: tp.Dict[int, Module] = {}
    module = _build_module(moduledef, index_module)
    state = _reref_state(state)

    for path, value in state.items():
        _set_value_at_path(module, path, value)

    return module


def _set_value_at_path(module: M, path: Path, value: tp.Any) -> M:
    if len(path) == 1:
        vars(module)[path[0]] = value
    else:
        _set_value_at_path(vars(module)[path[0]], path[1:], value)


def _reref_state(state: StateMapping) -> StateDict:
    new_state: StateDict = {}

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
    index_module[moduledef.index] = module

    return module


def _merge_state(states: tp.Iterable[StateMapping]) -> StateDict:
    new_state: StateDict = {}

    for state in states:
        new_state.update(state)

    return new_state


def _partition_by_collection(
    module: M,
) -> StateDef[tp.Dict[str, State], M]:
    state, moddef = module.deref()

    collections = tuple(
        set(x.collection for x in state.values() if isinstance(x, Value))
    )

    states = _split_state(state, *collections)

    if len(states[-1]) == 0:
        states = states[:-1]
    else:
        collections = (*collections, "rest")

    states = dict(zip(collections, states))

    return StateDef((states, moddef))  # type: ignore


def _split_state(
    state: State,
    *filters: partitioning.CollectionFilter,
) -> tp.Tuple[State, ...]:
    predicates = tuple(map(partitioning.to_predicate, filters))

    # we have n + 1 states, where n is the number of predicates
    # the last state is for values that don't match any predicate
    states: tp.Tuple[StateDict, ...] = tuple({} for _ in range(len(predicates) + 1))

    for path, value in state.items():
        for i, predicate in enumerate(predicates):
            if predicate(path, value):
                states[i][path] = value
                break
        else:
            # if we didn't break, set leaf to last state
            states[-1][path] = value

    return tuple(State(x) for x in states)


def _merge_partitions(states: tp.Iterable[State]) -> State:
    new_state: StateDict = {}

    for partition in states:
        new_state.update(partition)

    return State(new_state)
