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

from nnx import containers, nodes, reprlib

A = tp.TypeVar("A")
P = tp.TypeVar("P", bound="Pytree")


class TreeNode(containers.NodeBase[A]):
    pass


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
            assert isinstance(obj, Pytree)
            for field in dataclasses.fields(obj):
                if "nnx_container_fn" not in field.metadata:
                    continue

                container_fn = field.metadata["nnx_container_fn"]
                value = vars(obj)[field.name]
                value = container_fn(value)
                vars(obj)[field.name] = value

        vars(obj)["_pytree__sorted_fields"] = sorted(vars(obj))

        return obj


class Pytree(reprlib.Representable, metaclass=PytreeMeta):
    _pytree__is_mutable: bool
    _pytree__class_is_mutable: bool
    _pytree__sorted_fields: tp.Tuple[str, ...]

    if not tp.TYPE_CHECKING:

        def __getattribute__(self, name: str) -> tp.Any:
            value = object.__getattribute__(self, name)
            if isinstance(value, containers.Container):
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

        if name in vars_dict and isinstance(vars_dict[name], containers.Container):
            vars_dict[name] = vars_dict[name].replace(value=value)
        else:
            if isinstance(value, containers.Container):
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

            if nodes.is_node(value):
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
            if nodes.is_node(value)
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
            if not nodes.is_node(value):
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

    def __nnx_repr__(self):
        yield reprlib.Object(type(self))
        for name, value in vars(self).items():
            yield reprlib.Attr(name, repr(value))


# register node types
nodes.register_node_type(Pytree)
nodes.register_node_type(TreeNode)
