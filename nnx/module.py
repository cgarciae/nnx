import dataclasses
import typing as tp
from abc import ABCMeta
from typing import Any

import jax.tree_util as jtu

from nnx import containers, errors, ids, nodes, partitioning, reprlib, tracers
from nnx.containers import Container, Node, Sharding
from nnx.state import State

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="Module")
S = tp.TypeVar("S", bound=tp.Union[State, tp.Tuple[State, ...]])

Path = str
PathParts = tp.Tuple[str, ...]
StateDict = tp.Dict[Path, tp.Any]
StateMapping = tp.Mapping[Path, tp.Any]


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

    def __getitem__(self, key) -> "CallableProxy":
        return CallableProxy(self._proxy_context, self._proxy_callable[key])


def _identity(x):
    return x


@dataclasses.dataclass
class DelayedAccessor:
    accessor: tp.Callable[[tp.Any], tp.Any] = _identity

    def __call__(self, x):
        return self.accessor(x)

    def __getattr__(self, name):
        return DelayedAccessor(lambda x: getattr(x, name))

    def __getitem__(self, key):
        return DelayedAccessor(lambda x: x[key])


class ApplyCaller(tp.Protocol, tp.Generic[A]):
    def __getattr__(self, __name) -> "ApplyCaller[A]":
        ...

    def __getitem__(self, __name) -> "ApplyCaller[A]":
        ...

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, A]:
        ...


@dataclasses.dataclass(repr=False)
class _SubmodulesRepr(reprlib.Representable):
    submodules: tp.Tuple[tp.Tuple[str, tp.Union["ModuleDef[Module]", int]], ...]

    def __nnx_repr__(self):
        yield reprlib.Object(type="", value_sep=", ")

        for name, submodule in self.submodules:
            yield reprlib.Attr(repr(name), submodule, start="(", end=")")


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
        yield reprlib.Object(type=type(self))

        yield reprlib.Attr("type", self._type.__name__)
        yield reprlib.Attr("index", self._index)
        yield reprlib.Attr("submodules", _SubmodulesRepr(self._submodules))
        yield reprlib.Attr("static_fields", self._static_fields)

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

    def merge(self, state: State, *states: State) -> M:
        states = (state, *states)
        module = _build_module(self)
        current_state = State({})

        _update_module(module, current_state, states)

        return module

    def apply(self, state: State, *states: State) -> ApplyCaller["Pure[State, M]"]:
        accessesor = DelayedAccessor()

        def _context(accessesor, *args, **kwargs) -> tp.Tuple[tp.Any, Pure[State, M]]:
            module = self.merge(state, *states)
            fn = accessesor(module)
            out = fn(*args, **kwargs)
            return out, module.partition()

        return CallableProxy(_context, accessesor)  # type: ignore


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


class Pure(tp.Tuple[S, ModuleDef[M]]):
    @classmethod
    def new(cls, states: S, moduledef: ModuleDef[M]) -> "Pure[S, M]":
        return cls((states, moduledef))

    @property
    def states(self) -> S:
        return self[0]

    @property
    def moduledef(self) -> ModuleDef[M]:
        return self[1]

    def merge(self) -> M:
        if isinstance(self.states, tuple):
            return self.moduledef.merge(*self.states)
        else:
            return self.moduledef.merge(self.states)

    @property
    def apply(self) -> ApplyCaller["Pure[State, M]"]:
        if isinstance(self.states, tuple):
            return self.moduledef.apply(*self.states)
        else:
            return self.moduledef.apply(self.states)

    @property
    def call(self) -> M:
        accessesor = DelayedAccessor()

        def _context(accessesor, *args, **kwargs):
            module = self.merge()
            fn = accessesor(module)
            return fn(*args, **kwargs)

        return CallableProxy(_context, accessesor)  # type: ignore

    def get_state(self) -> State:
        if isinstance(self.states, tuple):
            return State.merge(*self.states)
        return self.states

    @tp.overload
    def filter(
        self,
        filter: partitioning.Filter,
        /,
    ) -> State:
        ...

    @tp.overload
    def filter(
        self,
        filter: partitioning.Filter,
        filter2: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> tp.Tuple[State, ...]:
        ...

    def filter(
        self, filter: partitioning.Filter, *filters: partitioning.Filter
    ) -> tp.Union[State, tp.Tuple[State, ...]]:
        filters = (filter, *filters)
        state = self.get_state()

        if len(filters) == 1:
            return state.filter(filters[0])
        else:
            return state.filter(filters[0], filters[1], *filters[2:])

    @tp.overload
    def partition(self) -> "Pure[State, M]":
        ...

    @tp.overload
    def partition(self, first: partitioning.Filter, /) -> "Pure[State, M]":
        ...

    @tp.overload
    def partition(
        self,
        first: partitioning.Filter,
        second: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> "Pure[tp.Tuple[State, ...], M]":
        ...

    def partition(
        self, *filters: partitioning.Filter
    ) -> tp.Union["Pure[State, M]", "Pure[tp.Tuple[State, ...], M]"]:
        state = self.get_state()

        if len(filters) == 0:
            states = state
        elif len(filters) == 1:
            states = state.partition(filters[0])
        else:
            states = state.partition(filters[0], filters[1], *filters[2:])

        if isinstance(states, State):
            return Pure.new(states, self.moduledef)
        else:
            return Pure.new(states, self.moduledef)

    @tp.overload
    def pop_state(
        self, filter: partitioning.Filter, /
    ) -> tp.Tuple[State, "Pure[State, M]"]:
        ...

    @tp.overload
    def pop_state(
        self,
        filter: partitioning.Filter,
        filter2: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> tp.Tuple[tp.Tuple[State, ...], "Pure[State, M]"]:
        ...

    def pop_state(
        self, filter: partitioning.Filter, *filters: partitioning.Filter
    ) -> tp.Tuple[tp.Union[State, tp.Tuple[State, ...]], "Pure[State, M]"]:
        filters = (filter, *filters)

        state = self.get_state()
        *states, rest = state.partition(*filters, ...)

        if len(states) == 1:
            states = states[0]
        else:
            states = tuple(states)

        return states, Pure.new(rest, self.moduledef)

    def update_state(
        self,
        updates: tp.Union[M, "Pure[S, M]", State, tp.Tuple[State, ...]],
    ) -> "Pure[State, M]":
        if isinstance(updates, Module):
            states = (updates.get_state(),)
        elif isinstance(updates, Pure):
            if isinstance(updates.states, tuple):
                states = updates.states
            elif isinstance(updates.states, State):
                states = (updates.states,)
            else:
                raise TypeError(
                    f"Expected Module, PureModule, State or tuple of State, "
                    f"got {type(updates.states).__name__}"
                )
        elif isinstance(updates, State):
            states = (updates,)
        elif isinstance(updates, tuple):
            states = updates
        else:
            raise TypeError(
                f"Expected Module, PureModule, State or tuple of State, "
                f"got {type(updates).__name__}"
            )

        if isinstance(self.states, tuple):
            states += self.states
        else:
            states += (self.states,)

        state = State.merge(*states)
        return Pure.new(state, self.moduledef)


def _pure_module_flatten(bounded: Pure[S, M]):
    return tuple(bounded), None


def _pure_module_unflatten(_, values: tp.Tuple[S, ModuleDef[M]]):
    return Pure(values)


jtu.register_pytree_node(Pure, _pure_module_flatten, _pure_module_unflatten)

PureModule = Pure[tp.Union[State, tp.Tuple[State, ...]], M]


SEEN_MODULES_REPR: tp.Optional[tp.Set[ids.UUID]] = None


class ModuleState(reprlib.Representable):
    __slots__ = ("_trace_state", "_id")

    def __init__(self, trace_state: tracers.TraceState):
        self._trace_state = trace_state
        self._id = ids.uuid()

    @property
    def trace_state(self) -> tracers.TraceState:
        return self._trace_state

    @property
    def id(self) -> ids.UUID:
        return self._id

    def __nnx_repr__(self):
        yield reprlib.Object(type(self))
        yield reprlib.Attr("trace_state", self._trace_state)


class ModuleMeta(ABCMeta):
    if not tp.TYPE_CHECKING:

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self._meta_call(*args, **kwargs)

    def _meta_call(self: tp.Type[M], *args, **kwargs) -> M:
        module = self.__new__(self, *args, **kwargs)
        vars(module)["_module__state"] = ModuleState(tracers.TraceState())
        module.__init__(*args, **kwargs)

        if dataclasses.is_dataclass(module):
            assert isinstance(module, Module)
            for field in dataclasses.fields(module):
                if "nnx_container_fn" not in field.metadata:
                    continue

                container_fn = field.metadata["nnx_container_fn"]
                value = vars(module)[field.name]
                value = container_fn(value)
                vars(module)[field.name] = value

        return module


@tp.runtime_checkable
class HasUnboxFn(tp.Protocol):
    unbox_fn: tp.Callable[["Container[tp.Any]"], tp.Any]


class Module(reprlib.Representable, metaclass=ModuleMeta):
    if tp.TYPE_CHECKING:
        _module__state: ModuleState

    if not tp.TYPE_CHECKING:

        def __getattribute__(self, name: str) -> Any:
            value = object.__getattribute__(self, name)
            if isinstance(value, Container):
                if isinstance(value, HasUnboxFn):
                    return value.unbox_fn(value)
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
        if name in vars_dict and isinstance(vars_dict[name], Container):
            vars_dict[name] = vars_dict[name].replace(value=value)
        else:
            if isinstance(value, Container):
                value = value.copy()
            vars_dict[name] = value

    def __hash__(self) -> int:
        return hash(self._module__state.id)

    def __nnx_repr__(self):
        global SEEN_MODULES_REPR

        if SEEN_MODULES_REPR is None:
            SEEN_MODULES_REPR = set()
            clear_seen = True
        else:
            clear_seen = False

        if self._module__state.id in SEEN_MODULES_REPR:
            yield reprlib.Object(type=type(self), empty_repr="...")
            return

        yield reprlib.Object(type=type(self))
        SEEN_MODULES_REPR.add(self._module__state.id)

        try:
            for name, value in vars(self).items():
                if isinstance(value, Module) or (
                    not nodes.is_node(value) and not name.startswith("_")
                ):
                    yield reprlib.Attr(name, value)
        finally:
            if clear_seen:
                SEEN_MODULES_REPR = None

    def clone(self: M) -> M:
        return self.partition().merge()

    @tp.overload
    def partition(self: M) -> Pure[State, M]:
        ...

    @tp.overload
    def partition(self: M, first: partitioning.Filter, /) -> Pure[State, M]:
        ...

    @tp.overload
    def partition(
        self: M,
        first: partitioning.Filter,
        second: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> Pure[tp.Tuple[State, ...], M]:
        ...

    def partition(
        self: M, *filters: partitioning.Filter
    ) -> tp.Union[Pure[State, M], Pure[tp.Tuple[State, ...], M]]:
        moduledef = _get_module_def(self)
        state = _get_module_state(self)

        if len(filters) == 0:
            states = state
        elif len(filters) == 1:
            states = state.partition(filters[0])
        else:
            states = state.partition(filters[0], filters[1], *filters[2:])

        if isinstance(states, tuple):
            return Pure.new(states, moduledef)
        else:
            return Pure.new(states, moduledef)

    def get_state(self) -> State:
        return _get_module_state(self)

    def get_module_def(self: M) -> ModuleDef[M]:
        return _get_module_def(self)

    @tp.overload
    def filter(self, first: partitioning.Filter, /) -> State:
        ...

    @tp.overload
    def filter(
        self,
        first: partitioning.Filter,
        second: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> tp.Tuple[State, ...]:
        ...

    def filter(
        self,
        first: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> tp.Union[State, tp.Tuple[State, ...]]:
        state = _get_module_state(self)

        if len(filters) == 0:
            states = state.filter(first)
        else:
            states = state.filter(first, filters[0], *filters[1:])

        return states

    @tp.overload
    def pop_state(
        self,
        filter: partitioning.Filter,
        /,
    ) -> State:
        ...

    @tp.overload
    def pop_state(
        self,
        filter: partitioning.Filter,
        filter2: partitioning.Filter,
        /,
        *filters: partitioning.Filter,
    ) -> tp.Tuple[State, ...]:
        ...

    def pop_state(
        self, *filters: partitioning.Filter
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
        accessesor = DelayedAccessor()

        def _context(accessesor, *args, **kwargs) -> tp.Tuple[tp.Any, M]:
            module = self.clone()
            fn = accessesor(module)
            out = fn(*args, **kwargs)
            return out, module

        return CallableProxy(_context, accessesor)  # type: ignore

    def update_state(
        self: M,
        updates: tp.Union[M, Pure[S, M], State, tp.Tuple[State, ...]],
    ) -> None:
        current_state = _get_module_state(self)

        if isinstance(updates, Pure):
            updates = updates.states
        elif isinstance(updates, Module):
            assert type(self) == type(updates)
            updates = _get_module_state(updates)

        _update_module(self, current_state, updates)

    def mutable_state_dict(self) -> tp.Dict[Path, "MutableLeaf"]:
        return {path: MutableLeaf(self, path) for path, _ in _iter_state(self)}

    def sow(self, collection: str, name: str, value: tp.Any) -> None:
        if hasattr(self, name):
            variable = vars(self)[name]
            if not isinstance(variable, containers.Variable):
                raise ValueError(
                    f"Expected '{name}' to be a Variable, got {type(variable).__name__}"
                )
            elif variable.collection != collection:
                raise ValueError(
                    f"Expected '{name}' to be in collection '{collection}', "
                    f"got '{variable.collection}'"
                )
            current_value = variable.value
            if not isinstance(current_value, tuple):
                raise ValueError(
                    f"Expected '{name}' to be a tuple, "
                    f"got {type(current_value).__name__}"
                )
            value = current_value + (value,)
            setattr(self, name, value)
        else:
            setattr(self, name, containers.variable(collection, (value,)))

    def for_each(self, module_type: tp.Type[M], fn: tp.Callable[[M], None]) -> None:
        visited: tp.Set[ids.UUID] = set()
        self._on_all(module_type, fn, visited)

    def _on_all(
        self,
        module_type: tp.Type[M],
        fn: tp.Callable[[M], None],
        visited: tp.Set[ids.UUID],
    ) -> None:
        if self._module__state.id in visited:
            return

        visited.add(self._module__state.id)

        if isinstance(self, module_type):
            fn(self)

        for value in vars(self).values():
            if isinstance(value, Module):
                value._on_all(module_type, fn, visited)

    # Pytree Definition
    # def __init_subclass__(cls) -> None:
    #     super().__init_subclass__()

    #     def _flatten(module: Module, *, with_keys: bool):
    #         state, moduledef = module.partition()
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


class MutableLeaf(reprlib.Representable):
    __slots__ = ("_module", "_name")

    def __init__(self, root: Module, path: Path):
        path_parts = path.split("/")
        *module_path, name = path_parts
        module = _get_value_path(root, module_path)
        if not isinstance(module, Module):
            raise ValueError(
                f"Expected a module at path {path_parts[:-1]}, ",
                f" got {type(module).__name__}",
            )
        self._module = module
        self._name = name

    @property
    def value(self) -> tp.Any:
        return getattr(self._module, self._name)

    @value.setter
    def value(self, value: tp.Any) -> None:
        setattr(self._module, self._name, value)

    @property
    def collection(self) -> tp.Optional[str]:
        attr = vars(self._module)[self._name]
        if not isinstance(attr, containers.Variable):
            return None

        return attr.collection

    @property
    def sharding(self) -> tp.Optional[Sharding]:
        attr = vars(self._module)[self._name]
        if not isinstance(attr, Node):
            return None

        return attr.metadata.get("sharding", None)


def _get_module_state(module: Module) -> State:
    return State(_iter_state(module))


def _get_module_def(module: M) -> ModuleDef[M]:
    module_index: tp.Dict[ids.UUID, int] = {}
    path: PathParts = ()

    moduledef = _make_module_def_recursive(module, module_index, path)
    assert isinstance(moduledef, ModuleDef)

    return moduledef


def _make_module_def_recursive(
    module: M,
    module_index: tp.Dict[ids.UUID, int],
    path: PathParts,
) -> tp.Union[ModuleDef[M], int]:
    if module._module__state.id in module_index:
        return module_index[module._module__state.id]

    index = len(module_index)
    module_index[module._module__state.id] = index

    submodules = []
    static_fields = []

    for name, value in sorted(vars(module).items(), key=lambda x: x[0]):
        value_path = (*path, name)
        if isinstance(value, Module):
            submodule_def = _make_module_def_recursive(value, module_index, value_path)
            submodules.append((name, submodule_def))
        elif not nodes.is_node(value) and not name.startswith("_module__"):
            static_fields.append((name, value))

    module_def = ModuleDef(
        type=type(module),
        index=index,
        submodules=tuple(submodules),
        static_fields=tuple(static_fields),
    )
    return module_def


def _iter_state(module: Module) -> tp.Iterator[tp.Tuple[Path, tp.Any]]:
    seen_modules: tp.Set[ids.UUID] = set()
    path_parts: PathParts = ()

    yield from _iter_state_recursive(module, seen_modules, path_parts)


def _iter_state_recursive(
    module: Module, seen_modules: tp.Set[ids.UUID], path_parts: PathParts
) -> tp.Iterator[tp.Tuple[Path, tp.Any]]:
    if module._module__state.id in seen_modules:
        return

    seen_modules.add(module._module__state.id)

    for name, value in sorted(vars(module).items(), key=lambda x: x[0]):
        new_path_parts = (*path_parts, name)
        if isinstance(value, Module):
            yield from _iter_state_recursive(value, seen_modules, new_path_parts)
        elif nodes.is_node(value):
            path = "/".join(new_path_parts)
            yield path, value


def _set_value_at_path(
    module: tp.Any, path_parts: tp.Union[PathParts, tp.List[str]], value: tp.Any
):
    if len(path_parts) == 1:
        setattr(module, path_parts[0], value)
    else:
        _set_value_at_path(vars(module)[path_parts[0]], path_parts[1:], value)


def _get_value_path(module: tp.Any, path: tp.Sequence[str]) -> tp.Any:
    if len(path) == 0:
        return module
    else:
        return _get_value_path(vars(module)[path[0]], path[1:])


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
    filters: tp.Tuple[partitioning.Filter, ...],
) -> tp.Tuple[State, ...]:
    module_index: tp.Dict[ids.UUID, int] = {}
    path_parts: PathParts = ()
    predicates = tuple(partitioning.to_predicate(filter) for filter in filters)
    states = tuple({} for _ in predicates)
    _pop_recursive(module, module_index, path_parts, states, predicates)

    return tuple(State(x) for x in states)


def _pop_recursive(
    module: Module,
    module_index: tp.Dict[ids.UUID, int],
    path_parts: PathParts,
    states: tp.Tuple[tp.Dict[Path, tp.Any]],
    predicates: tp.Tuple[partitioning.Predicate, ...],
) -> None:
    if module._module__state.id in module_index:
        return

    for name, value in list(vars(module).items()):
        if isinstance(value, Module):
            _pop_recursive(value, module_index, (*path_parts, name), states, predicates)
            continue
        elif not nodes.is_node(value):
            continue

        path = "/".join((*path_parts, name))
        for state, predicate in zip(states, predicates):
            if predicate(path, value):
                state[path] = value
                delattr(module, name)
                break

    module_index[module._module__state.id] = len(module_index)


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
        path_parts = path.split("/")
        _set_value_at_path(module, path_parts, value)


def first_from(*args: tp.Optional[A]) -> A:
    """Return the first non-None argument."""
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError("No non-None arguments found.")


# register nodes
nodes.register_node_type(Module)
nodes.register_node_type(Pure)
