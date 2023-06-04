import dataclasses
from functools import partial
import typing as tp
from abc import ABC, ABCMeta
from typing import Any

import jax.tree_util as jtu

from nnx import errors, partitioning, tracers
from nnx.nodes import is_node_type, register_node_type
from nnx.state import State, Variable
from nnx import reprlib
import nnx

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="Module")

Path = tp.Tuple[str, ...]
StateDict = tp.Dict[Path, tp.Any]
StateMapping = tp.Mapping[Path, tp.Any]


class ApplyCaller(tp.Protocol, tp.Generic[A]):
    def __getattr__(self, __name) -> "ApplyCaller[A]":
        ...

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, A]:
        ...


class ModuleDef(tp.Generic[M], reprlib.Representable):
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

    def __nnx_repr__(self):
        yield reprlib.Config(type=f"{type(self).__name__}")

        yield reprlib.Elem("type", self._type.__name__)
        yield reprlib.Elem("index", repr(self._index))
        yield reprlib.Elem("submodules", repr(self._submodules))
        yield reprlib.Elem("static_fields", repr(self._static_fields))

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

    @tp.overload
    def merge(self, state: State, *states: State) -> M:
        ...

    def merge(self, *states: State) -> M:
        module = _build_module(self)
        current_state = State({})

        _update_module(module, current_state, states)

        return module

    def apply(self, state: State, *states: State) -> ApplyCaller["PureModule[M]"]:
        module = self.merge(state, *states)

        def _context(fn, *args, **kwargs) -> tp.Tuple[tp.Any, PureModule[M]]:
            out = fn(*args, **kwargs)
            return out, module.split()

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


class PureModule(tp.Tuple[State, ModuleDef[M]]):
    @classmethod
    def new(cls, state: State, moduledef: ModuleDef[M]) -> "PureModule[M]":
        return cls((state, moduledef))

    @property
    def state(self) -> State:
        return self[0]

    @property
    def moduledef(self) -> ModuleDef[M]:
        return self[1]

    def merge(self) -> M:
        return self.moduledef.merge(self.state)

    @property
    def apply(self) -> ApplyCaller["PureModule[M]"]:
        return self.moduledef.apply(self.state)

    @tp.overload
    def get_state(
        self,
        filter: partitioning.CollectionFilter,
        /,
    ) -> State:
        ...

    @tp.overload
    def get_state(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[State, ...]:
        ...

    def get_state(
        self, *filters: partitioning.CollectionFilter
    ) -> tp.Union[State, tp.Tuple[State, ...]]:
        return self.state.filter(*filters)

    @tp.overload
    def split(self, first: partitioning.CollectionFilter, /) -> "PureModule[M]":
        ...

    @tp.overload
    def split(
        self,
        first: partitioning.CollectionFilter,
        second: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[tp.Tuple[State, ...], ModuleDef[M]]:
        ...

    def split(
        self,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Union["PureModule[M]", tp.Tuple[tp.Tuple[State, ...], ModuleDef[M]],]:
        states = self.state.split(*filters)
        if isinstance(states, State):
            return PureModule.new(states, self.moduledef)
        else:
            return states, self.moduledef

    @tp.overload
    def pop_state(
        self, filter: partitioning.CollectionFilter, /
    ) -> tp.Tuple[State, "PureModule[M]"]:
        ...

    @tp.overload
    def pop_state(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[tp.Tuple[State, ...], "PureModule[M]"]:
        ...

    def pop_state(
        self, *filters: partitioning.CollectionFilter
    ) -> tp.Tuple[tp.Union[State, tp.Tuple[State, ...]], "PureModule[M]"]:
        if len(filters) == 0:
            raise ValueError("At least one filter must be provided")
        else:
            *states, rest = self.state.split(*filters)

        if len(states) == 1:
            states = states[0]
        else:
            states = tuple(states)

        return states, PureModule.new(rest, self.moduledef)

    def update_state(
        self,
        updates: tp.Union[M, "PureModule[M]", State, tp.Tuple[State, ...]],
    ) -> "PureModule[M]":
        if isinstance(updates, Module):
            states = (updates.split()[0],)
        elif isinstance(updates, PureModule):
            states = (updates.state,)
        elif isinstance(updates, State):
            states = (updates,)
        elif isinstance(updates, tuple):
            states = updates
        else:
            raise TypeError(
                f"Expected Module, PureModule, State or tuple of State, "
                f"got {type(updates).__name__}"
            )

        state = State.merge(*states)
        return PureModule.new(state, self.moduledef)


def _pure_module_flatten(bounded: PureModule[M]):
    return tuple(bounded), None


def _pure_module_unflatten(_, values: tp.Tuple[State, ModuleDef[M]]):
    return PureModule(values)


jtu.register_pytree_node(PureModule, _pure_module_flatten, _pure_module_unflatten)


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


SEEN_MODULES_REPR: tp.Set[int] = set()


class ModuleState(reprlib.Representable):
    __slots__ = ("_trace_state",)

    def __init__(self, trace_state: tracers.TraceState):
        self._trace_state = trace_state

    @property
    def trace_state(self) -> tracers.TraceState:
        return self._trace_state

    def __nnx_repr__(self):
        yield reprlib.Config(f"{type(self).__name__}")
        yield reprlib.Elem("trace_state", repr(self._trace_state))


class ModuleMeta(ABCMeta):
    if not tp.TYPE_CHECKING:

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.call(*args, **kwargs)

    def call(self: tp.Type[M], *args, **kwargs) -> M:
        module = self.__new__(self, *args, **kwargs)
        vars(module)["_module__state"] = ModuleState(tracers.TraceState())
        module.__init__(*args, **kwargs)
        return module


class Module(reprlib.Representable, metaclass=ModuleMeta):
    if tp.TYPE_CHECKING:
        _module__state: ModuleState

    if not tp.TYPE_CHECKING:

        def __getattribute__(self, name: str) -> Any:
            value = object.__getattribute__(self, name)
            if isinstance(value, Variable):
                return value.value
            return value

        def __setattr__(self, name: str, value: Any) -> None:
            self._setattr(name, value)

    def _setattr(self, name: str, value: Any) -> None:
        if not self._module__state.trace_state.is_valid():
            raise errors.TraceContextError(
                "Cannot mutate Module from different trace level"
            )

        vars_dict = vars(self)
        if (
            name in vars_dict
            and isinstance(vars_dict[name], Variable)
            and not isinstance(value, Variable)
        ):
            vars_dict[name] = vars_dict[name].replace(value=value)
        else:
            if isinstance(value, Variable):
                value = value.copy()
            vars_dict[name] = value

    def __hash__(self) -> int:
        return id(self)

    def __nnx_repr__(self):
        if id(self) in SEEN_MODULES_REPR:
            yield reprlib.Config(type=f"{type(self).__name__}", empty_repr="...")
            return

        yield reprlib.Config(type=f"{type(self).__name__}")
        SEEN_MODULES_REPR.add(id(self))

        try:
            for name, value in vars(self).items():
                if isinstance(value, Module) or not is_node_type(value):
                    yield reprlib.Elem(name, repr(value))
        finally:
            SEEN_MODULES_REPR.remove(id(self))

    def clone(self: M) -> M:
        return self.split().merge()

    @tp.overload
    def split(self: M) -> PureModule[M]:
        ...

    @tp.overload
    def split(self: M, first: partitioning.CollectionFilter, /) -> PureModule[M]:
        ...

    @tp.overload
    def split(
        self: M,
        first: partitioning.CollectionFilter,
        second: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[tp.Tuple[State, ...], ModuleDef[M]]:
        ...

    def split(
        self: M,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Union[PureModule[M], tp.Tuple[tp.Tuple[State, ...], ModuleDef[M]]]:
        moduledef = _get_module_def(self)
        state = _get_module_state(self)

        if len(filters) == 0:
            states = state
        else:
            *states, rest = state.split(*filters)

            if rest:
                raise ValueError(
                    f"Non-exhaustive filters, got a non-empty remainder: "
                    f"{list(rest.keys())}.\nUse `...` to match all remaining elements."
                )

            if len(states) == 1:
                states = states[0]
            else:
                states = tuple(states)

        if isinstance(states, tuple):
            return states, moduledef
        else:
            return PureModule.new(states, moduledef)

    @tp.overload
    def get_state(self) -> State:
        ...

    @tp.overload
    def get_state(self, filter: partitioning.CollectionFilter, /) -> State:
        ...

    @tp.overload
    def get_state(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[State, ...]:
        ...

    def get_state(
        self, *filters: partitioning.CollectionFilter
    ) -> tp.Union[State, tp.Tuple[State, ...]]:
        state = _get_module_state(self)
        if len(filters) == 0:
            return state
        return state.filter(*filters)

    @tp.overload
    def pop_state(
        self,
        filter: partitioning.CollectionFilter,
        /,
    ) -> State:
        ...

    @tp.overload
    def pop_state(
        self,
        filter: partitioning.CollectionFilter,
        filter2: partitioning.CollectionFilter,
        /,
        *filters: partitioning.CollectionFilter,
    ) -> tp.Tuple[State, ...]:
        ...

    def pop_state(
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
    def apply(self: M) -> ApplyCaller[M]:
        module = self.clone()

        def _context(fn, *args, **kwargs) -> tp.Tuple[tp.Any, M]:
            out = fn(*args, **kwargs)
            return out, module

        return CallableProxy(_context, module)  # type: ignore

    def update_state(
        self: M,
        updates: tp.Union[M, PureModule[M], State, tp.Tuple[State, ...]],
    ) -> None:
        current_state = _get_module_state(self)

        if isinstance(updates, PureModule):
            updates = updates.state
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
            state = {sep.join(path): leaf for path, leaf in _iter_state(self)}
        else:
            state = {path: leaf for path, leaf in _iter_state(self)}

        return state

    # Pytree Definition
    # def __init_subclass__(cls) -> None:
    #     super().__init_subclass__()

    #     def _flatten(module: Module, *, with_keys: bool):
    #         state, moduledef = module.split()
    #         paths = tuple(state.keys())

    #         if with_keys:
    #             nodes = tuple(
    #                 (jtu.DictKey(path), value) for path, value in state.items()
    #             )
    #         else:
    #             nodes = tuple(state.values())

    #         return nodes, (paths, moduledef)

    #     def _unflatten(
    #         paths_moddef: tp.Tuple[tp.Tuple[Path, ...], ModuleDef[M]],
    #         nodes: tp.Tuple[tp.Any, ...],
    #     ) -> M:
    #         paths, moduledef = paths_moddef
    #         return moduledef.merge(State(zip(paths, nodes)))

    #     jtu.register_pytree_with_keys(
    #         cls,
    #         partial(_flatten, with_keys=True),
    #         _unflatten,
    #         flatten_func=partial(_flatten, with_keys=False),
    #     )


def _get_module_state(module: Module) -> State:
    return State(_iter_state(module))


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

    index = len(module_index)
    module_index[id(module)] = index

    submodules = []
    static_fields = []

    for name, value in sorted(vars(module).items(), key=lambda x: x[0]):
        value_path = (*path, name)
        if isinstance(value, Module):
            submodule_def = _make_module_def_recursive(value, module_index, value_path)
            submodules.append((name, submodule_def))
        elif not nnx.is_node_type(value) and not name.startswith("_module__"):
            static_fields.append((name, value))

    module_def = ModuleDef(
        type=type(module),
        index=index,
        submodules=tuple(submodules),
        static_fields=tuple(static_fields),
    )
    return module_def


def _iter_state(module: Module) -> tp.Iterator[tp.Tuple[Path, tp.Any]]:
    seen_modules: tp.Set[int] = set()
    path: Path = ()

    yield from _iter_state_recursive(module, seen_modules, path)


def _iter_state_recursive(
    module: Module, seen_modules: tp.Set[int], path: Path
) -> tp.Iterator[tp.Tuple[Path, tp.Any]]:
    if id(module) in seen_modules:
        return

    seen_modules.add(id(module))

    for name, value in sorted(vars(module).items(), key=lambda x: x[0]):
        value_path = (*path, name)
        if isinstance(value, Module):
            yield from _iter_state_recursive(value, seen_modules, value_path)
        elif nnx.is_node_type(value):
            yield value_path, value


def _set_value_at_path(module: M, path: Path, value: tp.Any) -> M:
    if len(path) == 1:
        vars(module)[path[0]] = value
    else:
        _set_value_at_path(vars(module)[path[0]], path[1:], value)


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

    # add a dummy module to the index to avoid infinite recursion
    module = object.__new__(moduledef.type)
    index_module[moduledef.index] = module

    submodules = {
        name: _build_module_recursive(submodule, index_module)
        for name, submodule in moduledef.submodules
    }

    vars(module).update(moduledef.static_fields)
    vars(module).update(submodules)
    vars(module)["_module__state"] = ModuleState(tracers.TraceState())

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
        elif not nnx.is_node_type(value):
            continue

        for state, predicate in zip(states, predicates):
            if predicate(value_path, value):
                state[value_path] = value
                delattr(module, name)
                break

    module_index[id(module)] = len(module_index)


def _update_module(
    module: Module, current_state: State, updates: tp.Union[State, tp.Tuple[State, ...]]
) -> None:
    new_states = [current_state]

    if isinstance(updates, State):
        new_states.append(updates)
    else:
        new_states.extend(updates)

    state: StateDict = {}
    for new_state in new_states:
        state.update(new_state)

    for path, value in state.items():
        _set_value_at_path(module, path, value)


# register nodes
register_node_type(Module)
register_node_type(PureModule)
