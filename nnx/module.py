import dataclasses
from functools import partial
import typing as tp
from abc import ABC
from typing import Any

import jax.tree_util as jtu

from nnx import partitioning, tracers
from nnx.state import State, Constant, Variable
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
        module = _build_module(self)
        current_state = State({})

        _update_module(module, current_state, states)

        return module

    def apply(
        self,
        states: tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
    ) -> ApplyCaller["Split[State, M]"]:
        module = self.merge(states)

        def _context(fn, *args, **kwargs) -> tp.Tuple[tp.Any, Split[State, M]]:
            out = fn(*args, **kwargs)
            return out, module.split(...)

        return CallableProxy(_context, module)  # type: ignore


def _moddef_flatten(moduledef: ModuleDef[M]):
    return (), (
        moduledef._type,
        moduledef._index,
        moduledef._submodules,
        moduledef._static_fields,
    )


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

    @tp.overload
    def pop(
        self,
        filter: partitioning.CollectionFilter,
        /,
    ) -> tp.Tuple[State, "Split[State, M]"]:
        ...

    @tp.overload
    def pop(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[tp.Tuple[State, ...], "Split[State, M]"]:
        ...

    def pop(
        self, *filters: partitioning.CollectionFilter
    ) -> tp.Tuple[tp.Union[State, tp.Tuple[State, ...]], "Split[State, M]"]:
        module = self.merge()
        states = module.pop(*filters)
        return states, module.split(...)

    def update(
        self,
        updates: tp.Union[
            M, "AnySplit[M]", State, tp.Tuple[State, ...], tp.Dict[str, State]
        ],
    ) -> "Split[State, M]":
        module = self.merge()
        module.update(updates)
        return module.split(...)


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
        moduledef = _get_module_def(self)
        state = _get_module_state(self)
        states = state.split(*filters)
        return Split((states, moduledef))

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

        state = _get_module_state(self)
        return state.get(*filters)

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
        [tp.Union[M, AnySplit[M], State, tp.Tuple[State, ...], tp.Dict[str, State]]],
        None,
    ]:
        return lambda states: self._update(states)

    @update.setter
    def update(
        self: M,
        value: tp.Union[
            M, AnySplit[M], State, tp.Tuple[State, ...], tp.Dict[str, State]
        ],
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
        updates: tp.Union[
            M, AnySplit[M], State, tp.Tuple[State, ...], tp.Dict[str, State]
        ],
    ) -> None:
        current_state = _get_module_state(self)

        if isinstance(updates, Split):
            updates = updates.states
        elif isinstance(updates, Module):
            assert type(self) == type(updates)
            updates = _get_module_state(updates)

        _update_module(self, current_state, updates)

    @tp.overload
    def state_dict(self) -> tp.Dict[Path, tp.Any]:
        ...

    @tp.overload
    def state_dict(self, sep: str) -> tp.Dict[str, tp.Any]:
        ...

    def state_dict(
        self, sep: tp.Optional[str] = None
    ) -> tp.Union[tp.Dict[Path, tp.Any], tp.Dict[str, tp.Any]]:
        if sep is not None:
            state = {
                sep.join(path): leaf for path, leaf in _iter_state(self, as_const=False)
            }
        else:
            state = {path: leaf for path, leaf in _iter_state(self, as_const=False)}

        return state

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        def _flatten(module: Module, *, with_keys: bool):
            state, moduledef = module.split(...)
            paths = tuple(state.keys())

            if with_keys:
                nodes = tuple(
                    (jtu.DictKey(path), value) for path, value in state.items()
                )
            else:
                nodes = tuple(state.values())

            return nodes, (paths, moduledef)

        def _unflatten(
            paths_moddef: tp.Tuple[tp.Tuple[Path, ...], ModuleDef[M]],
            nodes: tp.Tuple[tp.Any, ...],
        ) -> M:
            paths, moduledef = paths_moddef
            return moduledef.merge(State(zip(paths, nodes)))

        jtu.register_pytree_with_keys(
            cls,
            partial(_flatten, with_keys=True),
            _unflatten,
            flatten_func=partial(_flatten, with_keys=False),
        )


def _get_module_state(module: Module) -> State:
    return State(_iter_state(module, as_const=True))


def _get_module_def(module: M) -> ModuleDef[M]:
    module_index: tp.Dict[int, int] = {}
    path: Path = ()

    moduledef = _make_module_def_recursive(module, module_index, path)
    assert isinstance(moduledef, ModuleDef)

    return moduledef


def _make_module_def_recursive(
    module: M,
    module_index: tp.Dict[int, int],
    path: Path,
) -> tp.Union[ModuleDef[M], int]:
    if id(module) in module_index:
        return module_index[id(module)]

    submodules = []
    static_fields = []

    for name, value in sorted(vars(module).items(), key=lambda x: x[0]):
        value_path = (*path, name)
        if isinstance(value, Module):
            submodule_dag = _make_module_def_recursive(value, module_index, value_path)
            submodules.append((name, submodule_dag))
        elif not nnx.is_node_type(value):
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


def _iter_state(
    module: Module, *, as_const: bool
) -> tp.Iterator[tp.Tuple[Path, tp.Any]]:
    seen_modules: tp.Set[int] = set()
    path: Path = ()

    yield from _iter_state_recursive(module, seen_modules, path, as_const)


def _iter_state_recursive(
    module: Module,
    seen_modules: tp.Set[int],
    path: Path,
    as_const: bool,
) -> tp.Iterator[tp.Tuple[Path, tp.Any]]:
    if id(module) in seen_modules:
        return

    seen_modules.add(id(module))

    for name, value in sorted(vars(module).items(), key=lambda x: x[0]):
        value_path = (*path, name)
        if isinstance(value, Module):
            yield from _iter_state_recursive(value, seen_modules, value_path, as_const)
        elif nnx.is_node_type(value):
            if isinstance(value, Variable) and as_const:
                value = value.to_const()
            yield value_path, value


def _set_value_at_path(module: M, path: Path, value: tp.Any) -> M:
    if len(path) == 1:
        vars(module)[path[0]] = value
    else:
        _set_value_at_path(vars(module)[path[0]], path[1:], value)


def _consts_to_variables(state: StateMapping) -> StateDict:
    new_state: StateDict = {}
    context_trace = tracers.get_top_trace(state)

    for path, value in state.items():
        if isinstance(value, Constant):
            new_state[path] = value.to_var(context_trace)
        else:
            new_state[path] = value

    return new_state


def _build_module(moduledef: ModuleDef[M]) -> M:
    index_module: tp.Dict[int, Module] = {}
    module = _build_module_recursive(moduledef, index_module)
    return module


def _build_module_recursive(
    moduledef: tp.Union[ModuleDef[M], int],
    index_module: tp.Dict[int, Module],
) -> M:
    if isinstance(moduledef, int):
        return index_module[moduledef]  # type: ignore

    assert moduledef.index not in index_module

    submodules = {
        name: _build_module_recursive(submodule, index_module)
        for name, submodule in moduledef.submodules
    }

    module = object.__new__(moduledef.type)
    vars(module).update(moduledef.static_fields)
    vars(module).update(submodules)
    index_module[moduledef.index] = module

    return module


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
            value = value.to_const()
        elif not nnx.is_node_type(value):
            continue

        for state, predicate in zip(states, predicates):
            if predicate(value_path, value):
                state[value_path] = value
                delattr(module, name)
                break

    module_index[id(module)] = len(module_index)


def _update_module(
    module: Module,
    current_state: State,
    updates: tp.Union[State, tp.Tuple[State, ...], tp.Dict[str, State]],
) -> None:
    new_states = [current_state]

    if isinstance(updates, State):
        new_states.append(updates)
    elif isinstance(updates, dict):
        new_states.extend(updates.values())
    else:
        new_states.extend(updates)

    state = State.merge(tuple(new_states))
    state = _consts_to_variables(state)

    for path, value in state.items():
        _set_value_at_path(module, path, value)
