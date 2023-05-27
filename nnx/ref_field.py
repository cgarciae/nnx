import functools
import typing as tp
import dataclasses

import jax
from nnx.nn import initializers

from nnx.reference import Ref, Sharding


A = tp.TypeVar("A")
K = tp.TypeVar("K", bound=tp.Hashable)


@dataclasses.dataclass
class RefMetadata(tp.Generic[A]):
    value: A
    sharding: Sharding


def with_partitioning(
    initializer: initializers.Initializer,
    sharding: Sharding,
) -> initializers.Initializer:
    @functools.wraps(initializer)
    def wrapper(*args):
        return RefMetadata(initializer(*args), sharding)

    return wrapper  # type: ignore


def ref_metadata(value: A, sharding: Sharding) -> A:
    return RefMetadata(value, sharding)  # type: ignore


class RefField(dataclasses.Field, tp.Generic[A]):
    def __init__(
        self,
        *,
        collection: str = "",
        default: tp.Any = dataclasses.MISSING,
        default_factory: tp.Any = dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
        metadata: tp.Optional[tp.Mapping[tp.Any, tp.Any]] = None,
    ):
        if metadata is None:
            metadata = {}
        super().__init__(default, default_factory, init, repr, hash, compare, metadata)
        self.collection = collection
        self._first_get_call = True
        self.class_field_name: tp.Optional[str] = None

    def __set_name__(self, cls, name):
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
                return

        if not hasattr(obj, self.object_field_name):
            raise AttributeError(f"Attribute '{self.class_field_name}' is not set")

        return getattr(obj, self.object_field_name).value

    def __set__(self, obj, value: tp.Union[A, RefMetadata[A]]):
        if isinstance(value, Ref):
            raise ValueError("Cannot change Ref")
        elif hasattr(obj, self.object_field_name):
            if isinstance(value, RefMetadata):
                raise ValueError("Cannot change RefMetadata after initialization")
            ref: Ref[A] = getattr(obj, self.object_field_name)
            ref.value = value
        else:
            if isinstance(value, RefMetadata):
                value, sharding = value.value, value.sharding
            else:
                sharding = None

            obj.__dict__[self.object_field_name] = Ref(
                value, collection=self.collection, sharding=sharding
            )
