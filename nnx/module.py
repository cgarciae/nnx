import dataclasses
from functools import partial
import typing as tp
from abc import ABC
from typing import Any

import jax.tree_util as jtu

from nnx import partitioning, tracers
from nnx.state import State, Value, Variable
import nnx

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="Module")
P = tp.TypeVar(
    "P",
    bound=tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
)
Path = tp.Tuple[str, ...]
StateDict = tp.Dict[Path, tp.Any]
StateMapping = tp.Mapping[Path, tp.Any]


class ApplyCaller(tp.Protocol, tp.Generic[A]):
    def __getattr__(self, __name) -> "ApplyCaller[A]":
        ...

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, A]:
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

    def merge(
        self,
        states: tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> M:
        if not isinstance(states, (State, tuple, dict)):
            raise TypeError(
                f"states must be a State, tuple of State, or dict of State, "
                f"got {type(states).__name__}"
            )
        if isinstance(states, State):
            state = states
        elif isinstance(states, tuple):
            state = _merge_state(states)
        else:
            state = _merge_state(states.values())

        return _merge(state, self)

    def apply(
        self,
        states: tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> ApplyCaller["Split[State, M]"]:
        module = self.merge(states)

        def _context(fn, *args, **kwargs) -> tp.Tuple[tp.Any, Split[State, M]]:
            out = fn(*args, **kwargs)
            return out, module.split(...)

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


class Split(tp.Tuple[P, ModuleDef[M]]):
    @property
    def states(self) -> P:
        return self[0]

    @property
    def moduledef(self) -> ModuleDef[M]:
        return self[1]

    def merge(self) -> M:
        return self.moduledef.merge(self.states)

    @property
    def apply(self) -> ApplyCaller["Split[State, M]"]:
        return self.moduledef.apply(self.states)

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
        return self.merge().get(*filters)

    @tp.overload
    def split(
        self,
        first: None = None,
        second: None = None,
        /,
    ) -> "Split[tp.Dict[str, State], M]":
        ...

    @tp.overload
    def split(
        self, first: partitioning.CollectionFilter, second: None = None, /
    ) -> "Split[State, M]":
        ...

    @tp.overload
    def split(
        self,
        first: partitioning.CollectionFilter,
        second: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> "Split[tp.Tuple[State, ...], M]":
        ...

    def split(
        self,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Union[
        "Split[State, M]",
        "Split[tp.Tuple[State, ...], M]",
        "Split[tp.Dict[str, State], M]",
    ]:
        return self.merge().split(*filters)


AnySplit = tp.Union[
    Split[State, M],
    Split[tp.Tuple[State, ...], M],
    Split[tp.Dict[str, State], M],
]


def _derefedmod_flatten(bounded: Split[P, M]):
    return tuple(bounded), None


def _derefedmod_unflatten(_, values):
    return Split(values)


jtu.register_pytree_node(Split, _derefedmod_flatten, _derefedmod_unflatten)


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

        def __getattribute__(self, name: str) -> Any:
            value = object.__getattribute__(self, name)
            if isinstance(value, Variable):
                return value.value
            return value

        def __setattr__(self, name: str, value: Any) -> None:
            self._setattr(name, value)

    def _setattr(self, name: str, value: Any) -> None:
        vars_dict = vars(self)
        if (
            name in vars_dict
            and isinstance(vars_dict[name], Variable)
            and not isinstance(value, Variable)
        ):
            vars_dict[name].value = value
        else:
            if isinstance(value, Variable):
                value = value.copy()
            object.__setattr__(self, name, value)

    def __hash__(self) -> int:
        return id(self)

    def deref(self: M) -> AnySplit[M]:
        state, moduledef = _deref(self)
        state = State(state)
        return Split((state, moduledef))

    def clone(self: M) -> M:
        return self.split(...).merge()

    @tp.overload
    def split(
        self: M,
        first: None = None,
        second: None = None,
        /,
    ) -> Split[tp.Dict[str, State], M]:
        ...

    @tp.overload
    def split(
        self: M, first: partitioning.CollectionFilter, second: None = None, /
    ) -> Split[State, M]:
        ...

    @tp.overload
    def split(
        self: M,
        first: partitioning.CollectionFilter,
        second: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> Split[tp.Tuple[State, ...], M]:
        ...

    def split(
        self: M,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Union[
        Split[State, M],
        Split[tp.Tuple[State, ...], M],
        Split[tp.Dict[str, State], M],
    ]:
        if len(filters) == 1 and filters[0] is Ellipsis:
            states, moddef = _deref(self)
            states = State(states)
        elif len(filters) == 0:
            states, moddef = _partition_by_collection(self)
        else:
            state, moddef = _deref(self)
            (*states, rest) = _split_state(state, *filters)

            if len(rest) > 0:
                raise ValueError(
                    f"Non-exhaustive filters, got a non-empty remainder: {rest}.\n"
                    f"Use `...` to match all remaining elements."
                )

            if len(states) == 1:
                states = states[0]
            else:
                states = tuple(states)

        return Split((states, moddef))  # type: ignore

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

        state, _ = _deref(self)
        (*states, _rest) = _split_state(state, *filters)

        assert len(states) == len(filters)

        if len(states) == 1:
            return states[0]
        else:
            return tuple(states)

    @tp.overload
    def pop(
        self,
        filter: partitioning.CollectionFilter,
        /,
    ) -> State:
        ...

    @tp.overload
    def pop(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[State, ...]:
        ...

    def pop(
        self, *filters: partitioning.CollectionFilter
    ) -> tp.Union[State, tp.Tuple[State, ...]]:
        if len(filters) == 0:
            raise ValueError("Expected at least one filter")

        states = _pop(self, filters)

        if len(states) == 1:
            return states[0]
        else:
            return states

    @property
    def update(
        self: M,
    ) -> tp.Callable[
        [tp.Union[M, State, tp.Tuple[State, ...], tp.Dict[str, State]]], None
    ]:
        return lambda states: self._update(states)

    @update.setter
    def update(
        self: M,
        value: tp.Union[M, State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> None:
        self._update(value)

    @property
    def apply(self: M) -> ApplyCaller[M]:
        module = self.clone()

        def _context(fn, *args, **kwargs) -> tp.Tuple[tp.Any, M]:
            out = fn(*args, **kwargs)
            return out, module

        return CallableProxy(_context, module)  # type: ignore

    def _update(
        self: M,
        states: tp.Union[M, State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> None:
        if isinstance(states, Module):
            assert type(self) == type(states)
            new_state, _ = _deref(states)
        elif isinstance(states, State):
            new_state = states
        else:
            if isinstance(states, dict):
                states = tuple(states.values())

            new_state = _merge_states(states)

        current_state = self.state_dict()
        context_trace = tracers.get_top_trace(
            [x.value if isinstance(x, Variable) else x for x in current_state.values()]
        )

        for path, new_value in new_state.items():
            if isinstance(new_value, Value):
                if path in current_state:
                    assert isinstance(current_state[path], Variable)
                    current_state[path].value = new_value.value
                else:
                    current_state[path] = new_value.to_var(context_trace)
                    _set_value_at_path(self, path, current_state[path])
            else:
                _set_value_at_path(self, path, new_value)

    @tp.overload
    def state_dict(self) -> tp.Dict[Path, tp.Any]:
        ...

    @tp.overload
    def state_dict(self, sep: str) -> tp.Dict[str, tp.Any]:
        ...

    def state_dict(
        self, sep: tp.Optional[str] = None
    ) -> tp.Union[tp.Dict[Path, tp.Any], tp.Dict[str, tp.Any]]:
        state = _state_dict(self)

        if sep is not None:
            state = {sep.join(path): leaf for path, leaf in state.items()}
        return state

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        def _flatten(module: Module, *, with_keys: bool):
            state, moddef = module.split(...)
            paths = tuple(state.keys())

            if with_keys:
                nodes = tuple(
                    (jtu.DictKey(path), value) for path, value in state.items()
                )
            else:
                nodes = tuple(state.values())

            return nodes, (paths, moddef)

        def _unflatten(
            paths_moddef: tp.Tuple[tp.Tuple[Path, ...], ModuleDef[M]],
            nodes: tp.Tuple[tp.Any, ...],
        ) -> M:
            paths, moddef = paths_moddef
            return moddef.merge(State(zip(paths, nodes)))

        jtu.register_pytree_with_keys(
            cls,
            partial(_flatten, with_keys=True),
            _unflatten,
            flatten_func=partial(_flatten, with_keys=False),
        )


def _deref(module: M) -> tp.Tuple[StateDict, ModuleDef[M]]:
    module_index: tp.Dict[int, int] = {}
    path: Path = ()
    state: tp.Dict[Path, tp.Any] = {}

    moduledef = _deref_recursive(module, module_index, path, state)
    assert isinstance(moduledef, ModuleDef)

    return state, moduledef


def _deref_recursive(
    module: M,
    module_index: tp.Dict[int, int],
    path: Path,
    state: tp.Dict[Path, tp.Any],
) -> tp.Union[ModuleDef[M], int]:
    if id(module) in module_index:
        return module_index[id(module)]

    submodules = []
    static_fields = []

    for name, value in vars(module).items():
        value_path = (*path, name)
        if isinstance(value, Module):
            submodule_dag = _deref_recursive(value, module_index, value_path, state)
            submodules.append((name, submodule_dag))
        elif isinstance(value, Variable):
            state[value_path] = value.to_value()
        elif nnx.is_node_type(value):
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
    module_index[id(module)] = index
    return module_dag


def _state_dict(module: Module) -> StateDict:
    seen_modules: tp.Set[int] = set()
    path: Path = ()
    state: StateDict = {}

    _state_dict_recursive(module, seen_modules, path, state)
    return state


def _state_dict_recursive(
    module: Module,
    seen_modules: tp.Set[int],
    path: Path,
    state: tp.Dict[Path, tp.Any],
) -> None:
    if id(module) in seen_modules:
        return

    seen_modules.add(id(module))

    for name, value in vars(module).items():
        value_path = (*path, name)
        if isinstance(value, Module):
            _state_dict_recursive(value, seen_modules, value_path, state)
        elif isinstance(value, Variable):
            state[value_path] = value
        elif nnx.is_node_type(value):
            state[value_path] = value


def _merge(state: StateMapping, moduledef: ModuleDef[M]) -> M:
    index_module: tp.Dict[int, Module] = {}
    module = _build_module(moduledef, index_module)
    state = _values_to_variables(state)

    for path, value in state.items():
        _set_value_at_path(module, path, value)

    return module


def _set_value_at_path(module: M, path: Path, value: tp.Any) -> M:
    if len(path) == 1:
        setattr(module, path[0], value)
    else:
        _set_value_at_path(vars(module)[path[0]], path[1:], value)


def _values_to_variables(state: StateMapping) -> StateDict:
    new_state: StateDict = {}
    context_trace = tracers.get_top_trace(state)

    for path, value in state.items():
        if isinstance(value, Value):
            new_state[path] = value.to_var(context_trace)
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


@tp.overload
def _partition_by_collection(module: M) -> Split[tp.Dict[str, State], M]:
    ...


@tp.overload
def _partition_by_collection(module: M, collection: str) -> Split[State, M]:
    ...


@tp.overload
def _partition_by_collection(
    module: M, collection: str, second: str, *rest: str
) -> Split[tp.Tuple[State, ...], M]:
    ...


def _partition_by_collection(
    module: M,
) -> Split[tp.Dict[str, State], M]:
    state, moddef = _deref(module)

    collections = tuple(
        set(x.collection for x in state.values() if isinstance(x, Value))
    )

    states = _split_state(state, *collections)

    if len(states[-1]) == 0:
        states = states[:-1]
    else:
        collections = (*collections, "rest")

    states = dict(zip(collections, states))

    return Split((states, moddef))  # type: ignore


def _split_state(
    state: StateMapping,
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


def _merge_states(
    states: tp.Iterable[StateMapping],
) -> State:
    new_state: StateDict = {}

    for state in states:
        new_state.update(state)

    return State(new_state)


def _pop(
    module: Module,
    filters: tp.Tuple[partitioning.CollectionFilter, ...],
) -> tp.Tuple[State, ...]:
    module_index: tp.Dict[int, int] = {}
    path: Path = ()
    predicates = tuple(partitioning.to_predicate(filter) for filter in filters)
    states = tuple({} for _ in predicates)
    _pop_recursive(module, module_index, path, states, predicates)

    return tuple(State(x) for x in states)


def _pop_recursive(
    module: Module,
    module_index: tp.Dict[int, int],
    path: Path,
    states: tp.Tuple[tp.Dict[Path, tp.Any]],
    predicates: tp.Tuple[partitioning.Predicate, ...],
) -> None:
    if id(module) in module_index:
        return

    for name, value in list(vars(module).items()):
        value_path = (*path, name)
        if isinstance(value, Module):
            _pop_recursive(value, module_index, value_path, states, predicates)
            continue
        elif isinstance(value, Variable):
            value = value.to_value()
        elif not nnx.is_node_type(value):
            continue

        for state, predicate in zip(states, predicates):
            if predicate(value_path, value):
                state[value_path] = value
                delattr(module, name)
                break

    module_index[id(module)] = len(module_index)
