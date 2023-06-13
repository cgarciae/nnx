import contextlib
import dataclasses
import importlib.util
import inspect
import typing as tp
from abc import ABCMeta
from copy import copy
from functools import partial
from types import MappingProxyType

import jax
import typing_extensions as tpe

from nnx import containers
from nnx.containers import Container
from nnx.nodes import is_node, register_node_type

# from nnx.ref_field import RefField

A = tp.TypeVar("A")
P = tp.TypeVar("P", bound="Pytree")


@contextlib.contextmanager
def _mutable(obj: P) -> tp.Iterator[None]:
    vars(obj)["_pytree__is_mutable"] = True
    try:
        yield
    finally:
        del vars(obj)["_pytree__is_mutable"]


@contextlib.contextmanager
def _initializing(obj: P) -> tp.Iterator[None]:
    vars(obj)["_pytree__initializing"] = True
    try:
        yield
    finally:
        del vars(obj)["_pytree__initializing"]


class PytreeMeta(ABCMeta):
    if not tp.TYPE_CHECKING:

        def __call__(cls: tp.Type[P], *args: tp.Any, **kwargs: tp.Any) -> P:
            return cls.call(*args, **kwargs)

    def call(cls: tp.Type[P], *args: tp.Any, **kwargs: tp.Any) -> P:
        obj: P = cls.__new__(cls, *args, **kwargs)
        vars(obj)["_pytree__sorted_fields"] = ["_pytree__sorted_fields"]

        with _mutable(obj), _initializing(obj):
            obj.__init__(*args, **kwargs)

            if dataclasses.is_dataclass(obj):
                for field in dataclasses.fields(obj):
                    if "pytree_node" not in field.metadata:
                        continue

                    value = getattr(obj, field.name)

                    if field.metadata["pytree_node"]:
                        value = containers.node(value)
                    else:
                        value = containers.static(value)

                    obj._setattr(field.name, value)

        vars(obj)["_pytree__sorted_fields"] = sorted(vars(obj))

        return obj


class Pytree(metaclass=PytreeMeta):
    _pytree__is_mutable: bool
    _pytree__class_is_mutable: bool
    _pytree__sorted_fields: tp.Tuple[str, ...]

    if not tp.TYPE_CHECKING:

        def __getattribute__(self, name: str) -> tp.Any:
            value = object.__getattribute__(self, name)
            if isinstance(value, Container):
                return value.value
            return value

        def __setattr__(self, name: str, value: tp.Any) -> None:
            self._setattr(name, value)

    def _setattr(self: P, name: str, value: tp.Any):
        vars_dict = vars(self)
        if "_pytree__initializing" in vars_dict:
            pass
        elif name not in vars_dict:
            raise AttributeError(r"Cannot add new fields to an initialized Pytree")
        elif (
            "_pytree__is_mutable" not in vars_dict
            and not self._pytree__class_is_mutable
        ):
            raise AttributeError(
                f"{type(self)} is immutable, trying to update field {name}"
            )

        if name in vars_dict and isinstance(vars_dict[name], Container):
            vars_dict[name] = vars_dict[name].replace_value(value)
        else:
            if isinstance(value, Container):
                value = value.copy()
            vars_dict[name] = value

    def __init_subclass__(cls, mutable: bool = False):
        super().__init_subclass__()
        # init class variables
        cls._pytree__is_mutable = False
        cls._pytree__class_is_mutable = mutable

        # TODO: clean up this in the future once minimal supported version is 0.4.7
        if hasattr(jax.tree_util, "register_pytree_with_keys"):
            if (
                "flatten_func"
                in inspect.signature(jax.tree_util.register_pytree_with_keys).parameters
            ):
                jax.tree_util.register_pytree_with_keys(
                    cls,
                    partial(
                        cls._pytree__flatten,
                        with_key_paths=True,
                    ),
                    cls._pytree__unflatten,
                    flatten_func=partial(
                        cls._pytree__flatten,
                        with_key_paths=False,
                    ),
                )
            else:
                jax.tree_util.register_pytree_with_keys(
                    cls,
                    partial(
                        cls._pytree__flatten,
                        with_key_paths=True,
                    ),
                    cls._pytree__unflatten,
                )
        else:
            jax.tree_util.register_pytree_node(
                cls,
                partial(
                    cls._pytree__flatten,
                    with_key_paths=False,
                ),
                cls._pytree__unflatten,
            )

        # flax serialization support
        if importlib.util.find_spec("flax") is not None:
            from flax import serialization

            serialization.register_serialization_state(
                cls, cls._to_flax_state_dict, cls._from_flax_state_dict
            )

    @classmethod
    def _pytree__flatten(
        cls,
        pytree: "Pytree",
        *,
        with_key_paths: bool,
    ):
        all_vars = vars(pytree)
        static = {}
        node_values = []
        node_names = []

        for field in pytree._pytree__sorted_fields:
            value = all_vars[field]

            if is_node(value):
                node_names.append(field)
                if with_key_paths:
                    node_values.append((jax.tree_util.GetAttrKey(field), value))
                else:
                    node_values.append(value)
            else:
                static[field] = value

        return node_values, (tuple(node_names), MappingProxyType(static))

    @classmethod
    def _pytree__unflatten(
        cls: tp.Type[P],
        metadata: tp.Tuple[tp.Tuple[str, ...], tp.Mapping[str, tp.Any]],
        node_values: tp.Tuple[tp.Any, ...],
    ) -> P:
        node_names, static_fields = metadata
        pytree = object.__new__(cls)
        pytree.__dict__.update(zip(node_names, node_values))
        pytree.__dict__.update(static_fields)
        return pytree

    @classmethod
    def _to_flax_state_dict(cls, pytree: "Pytree") -> tp.Dict[str, tp.Any]:
        from flax import serialization

        state_dict = {
            name: serialization.to_state_dict(getattr(pytree, name))
            for name, value in vars(pytree).items()
            if is_node(value)
        }
        return state_dict

    @classmethod
    def _from_flax_state_dict(
        cls,
        pytree: P,
        state: tp.Dict[str, tp.Any],
    ) -> P:
        """Restore the state of a data class."""
        from flax import serialization

        state = state.copy()  # copy the state so we can pop the restored fields.
        updates = {}
        for name, value in vars(pytree).items():
            if not is_node(value):
                continue
            if name not in state:
                raise ValueError(
                    f"Missing field {name} in state dict while restoring"
                    f" an instance of {type(pytree).__name__},"
                    f" at path {serialization.current_path()}"
                )
            value_state = state.pop(name)
            updates[name] = serialization.from_state_dict(value, value_state, name=name)
        if state:
            names = ",".join(state.keys())
            raise ValueError(
                f'Unknown field(s) "{names}" in state dict while'
                f" restoring an instance of {type(pytree).__name__}"
                f" at path {serialization.current_path()}"
            )
        return pytree.replace(**updates)

    def replace(self: P, **kwargs: tp.Any) -> P:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        if dataclasses.is_dataclass(self):
            return dataclasses.replace(self, **kwargs)

        unknown_keys = set(kwargs) - set(vars(self))
        if unknown_keys and not self._pytree__class_is_mutable:
            raise ValueError(
                f"Trying to replace unknown fields {unknown_keys} "
                f"for '{type(self).__name__}'"
            )

        pytree = copy(self)
        with _mutable(pytree):
            for key, value in kwargs.items():
                setattr(pytree, key, value)

        return pytree


# register node types
register_node_type(Pytree)

# ------------------------------------------
# dataclass
# ------------------------------------------


def field(
    *,
    default: tp.Any = dataclasses.MISSING,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
    return dataclasses.field(  # type: ignore
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def node_field(
    *,
    default: tp.Any = dataclasses.MISSING,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    if "pytree_node" in metadata:
        raise ValueError("'pytree_node' found in metadata")

    metadata["pytree_node"] = True

    return field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def static_field(
    *,
    default: tp.Any = dataclasses.MISSING,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    if "pytree_node" in metadata:
        raise ValueError("'pytree_node' found in metadata")

    metadata["pytree_node"] = False

    return field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


# def ref(
#     collection: str,
#     default: tp.Any = dataclasses.MISSING,
#     *,
#     default_factory: tp.Any = dataclasses.MISSING,
#     init: bool = True,
#     repr: bool = True,
#     hash: tp.Optional[bool] = None,
#     compare: bool = True,
#     metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
# ) -> tp.Any:
#     return RefField(
#         collection=collection,
#         default=default,
#         default_factory=default_factory,
#         init=init,
#         repr=repr,
#         hash=hash,
#         compare=compare,
#         metadata=metadata,
#     )


# def param(
#     default: tp.Any = dataclasses.MISSING,
#     *,
#     default_factory: tp.Any = dataclasses.MISSING,
#     init: bool = True,
#     repr: bool = True,
#     hash: tp.Optional[bool] = None,
#     compare: bool = True,
#     metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
# ) -> tp.Any:
#     return ref(
#         "params",
#         default=default,
#         default_factory=default_factory,
#         init=init,
#         repr=repr,
#         hash=hash,
#         compare=compare,
#         metadata=metadata,
#     )


@tp.overload
def dataclass(cls: tp.Type[A]) -> tp.Type[A]:
    ...


@tp.overload
def dataclass(
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> tp.Callable[[tp.Type[A]], tp.Type[A]]:
    ...


@tpe.dataclass_transform(field_specifiers=(field, node_field, static_field))
def dataclass(
    cls: tp.Optional[tp.Type[A]] = None,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> tp.Union[tp.Type[A], tp.Callable[[tp.Type[A]], tp.Type[A]]]:
    decorator = dataclasses.dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
    )

    if cls is None:
        return decorator

    return decorator(cls)
