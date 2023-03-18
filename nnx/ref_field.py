import typing as tp
import dataclasses
from refx.ref import Ref


A = tp.TypeVar("A")


class RefField(dataclasses.Field, tp.Generic[A]):
    def __init__(self, ref_type: tp.Type[Ref[A]], **kwargs):
        super().__init__(**kwargs)
        self._ref_type = ref_type
        self._first_get_call = True
        self.class_field_name: tp.Optional[str] = None

    def __set_name__(self, owner, name):
        """__set_name__ initializer for properties as per [PEP-487](https://peps.python.org/pep-0487/)"""
        self.class_field_name = name

    @property
    def object_field_name(self):
        return f"{self.class_field_name}__ref"

    def __get__(self, obj, objtype=None):
        if obj is None:
            if self._first_get_call:
                self._first_get_call = False
                return self
            else:
                raise AttributeError

        if not hasattr(obj, self.object_field_name):
            raise AttributeError(f"Attribute '{self.class_field_name}' is not set")

        return getattr(obj, self.object_field_name).value

    def __set__(self, obj, value):
        if isinstance(value, Ref):
            raise ValueError("Cannot change Ref")
        elif hasattr(obj, self.object_field_name):
            ref: Ref[A] = getattr(obj, self.object_field_name)
            ref.value = value
        else:
            obj.__dict__[self.object_field_name] = self._ref_type(value)


def ref_field(
    default: tp.Any = dataclasses.MISSING,
    *,
    ref_type: tp.Type[Ref[tp.Any]] = Ref,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    if "pytree_node" in metadata:
        raise ValueError("'pytree_node' found in metadata")

    metadata["pytree_node"] = True

    return RefField(
        ref_type=ref_type,
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )
