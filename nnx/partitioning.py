import builtins
import dataclasses
import typing as tp

import jax
import numpy as np

import nnx

if tp.TYPE_CHECKING:
    ellipsis = builtins.ellipsis
else:
    ellipsis = tp.Any

Path = tp.Tuple[str, ...]
Predicate = tp.Callable[[Path, tp.Any], bool]
FilterLiteral = tp.Union[str, type, Predicate, ellipsis, None]
Filter = tp.Union[FilterLiteral, tp.Tuple[FilterLiteral, ...]]


def to_predicate(filter: Filter) -> Predicate:
    if isinstance(filter, str):
        return FromCollection(filter)
    elif isinstance(filter, type):
        return OfType(filter)
    elif filter is Ellipsis:
        return Everything()
    elif filter is None:
        return Nothing()
    elif callable(filter):
        return filter
    elif isinstance(filter, tp.Tuple):
        return Any(*filter)
    else:
        raise TypeError(f"Invalid collection filter: {filter}")


@dataclasses.dataclass
class FromCollection:
    collection: str

    def __call__(self, path: Path, x: tp.Any):
        if isinstance(x, nnx.Variable):
            return x.collection == self.collection
        return False


@dataclasses.dataclass
class OfType:
    type: type

    def __call__(self, path: Path, x: tp.Any):
        return isinstance(x, self.type)


class Any:
    def __init__(self, *filters: Filter):
        self.predicates = tuple(
            to_predicate(collection_filter) for collection_filter in filters
        )

    def __call__(self, path: Path, x: tp.Any):
        return any(predicate(path, x) for predicate in self.predicates)


class All:
    def __init__(self, *filters: Filter):
        self.predicates = tuple(
            to_predicate(collection_filter) for collection_filter in filters
        )

    def __call__(self, path: Path, x: tp.Any):
        return all(predicate(path, x) for predicate in self.predicates)


class Not:
    def __init__(self, collection_filter: Filter):
        self.predicate = to_predicate(collection_filter)

    def __call__(self, path: Path, x: tp.Any):
        return not self.predicate(path, x)


class Everything:
    def __call__(self, path: Path, x: tp.Any):
        return True


class Nothing:
    def __call__(self, path: Path, x: tp.Any):
        return False


buffers = (jax.Array, np.ndarray)
