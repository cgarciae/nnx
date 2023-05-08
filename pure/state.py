from types import MappingProxyType
import typing as tp

import jax
import jax.tree_util as jtu


class Variable:
    __slots__ = ("value", "collection")

    def __init__(self, value: tp.Any, collection: str = "params"):
        self.value = value
        self.collection = collection

    @classmethod
    def from_value(cls, value: tp.Any) -> "Variable":
        return value if isinstance(value, Variable) else Variable(value)

    def copy(self) -> "Variable":
        return Variable(self.value, self.collection)


def _flatten_variable_with_keys(variable: Variable):
    node = (jtu.GetAttrKey("value"), variable.value)
    return (node,), variable.collection


def _flatten_variable(variable: Variable):
    return (variable.value,), variable.collection


def _unflatten_variable(collection: str, value: tp.Any):
    return Variable(value, collection)


jax.tree_util.register_pytree_with_keys(
    Variable,
    _flatten_variable_with_keys,
    _unflatten_variable,
    flatten_func=_flatten_variable,
)


# NOTE: Should this API be mutable or immutable?
# mutable is easier to use but easier to shoot yourself in the foot
# immutable is a bit more verbose but safer
class State(tp.MutableMapping[str, Variable]):
    __slots__ = ("_variables",)

    def __init__(self, *args, **kwargs: tp.Union[Variable, jax.Array]):
        self._variables = {
            k: Variable.from_value(v) for k, v in dict(*args, **kwargs).items()
        }

    def __getitem__(self, name: str) -> tp.Any:
        return self._variables[name].value

    def __setitem__(self, name: str, value: tp.Any):
        self._variables[name].value = value

    def __delitem__(self, name: str):
        del self._variables[name]

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __delattr__ = __delitem__

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self._variables)

    def keys(self) -> tp.KeysView[str]:
        return self._variables.keys()

    def values(self) -> tp.ValuesView[Variable]:
        return self._variables.values()

    def __repr__(self) -> str:
        return f"State({self._variables})"

    def _copy_variables(self) -> tp.Dict[str, Variable]:
        return {k: v.copy() for k, v in self._variables.items()}

    def copy(self) -> "State":
        return State((k, v.copy()) for k, v in self.items())

    def update(self, **kwargs: tp.Union[Variable, jax.Array]) -> "State":
        variables = self._copy_variables()
        variables.update((k, Variable.from_value(v)) for k, v in kwargs.items())
        return State(variables)

    @tp.overload
    def pop(self, name: str) -> tp.Tuple[tp.Any, "State"]:
        ...

    @tp.overload
    def pop(self, name: str, *names: str) -> tp.Tuple[tp.Tuple[tp.Any, ...], "State"]:
        ...

    def pop(
        self, *names: str
    ) -> tp.Tuple[tp.Union[tp.Any, tp.Tuple[tp.Any, ...]], "State"]:
        variables = self._copy_variables()

        if len(names) == 0:
            raise ValueError("pop expected at least 1 argument, got 0")
        elif len(names) == 1:
            name = names[0]
            value = variables.pop(name).value
            return value, State(variables)
        else:
            values = tuple(variables.pop(name).value for name in names)
            return values, State(variables)


def _state_flatten_with_keys(state: State):
    nodes = tuple((jtu.GetAttrKey(name), variable) for name, variable in state.items())
    names = tuple(state)
    return nodes, names


def _state_unflatten(names: tp.Tuple[str, ...], nodes: tp.Tuple[Variable, ...]):
    return State(zip(names, nodes))


def _state_flatten(state: State):
    return tuple(state.values()), tuple(state)


jtu.register_pytree_with_keys(
    State, _state_flatten_with_keys, _state_unflatten, flatten_func=_state_flatten
)
