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

FilterLiteral = tp.Union[str, type]
Path = tp.Tuple[str, ...]
Predicate = tp.Callable[[Path, tp.Any], bool]
CollectionFilter = tp.Union[
    FilterLiteral, tp.Sequence[FilterLiteral], Predicate, ellipsis
]


def to_predicate(collection_filter: CollectionFilter) -> Predicate:
    if isinstance(collection_filter, str):
        return FromCollection(collection_filter)
    elif isinstance(collection_filter, type):
        return OfType(collection_filter)
    elif collection_filter is Ellipsis:
        return Everything()
    elif callable(collection_filter):
        return collection_filter
    elif isinstance(collection_filter, tp.Sequence):
        return Any(collection_filter)
    else:
        raise TypeError(f"Invalid collection filter: {collection_filter}")


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
    def __init__(self, collection_filters: tp.Sequence[CollectionFilter]):
        self.predicates = tuple(
            to_predicate(collection_filter) for collection_filter in collection_filters
        )

    def __call__(self, path: Path, x: tp.Any):
        return any(predicate(path, x) for predicate in self.predicates)


class Not:
    def __init__(self, collection_filter: CollectionFilter):
        self.predicate = to_predicate(collection_filter)

    def __call__(self, path: Path, x: tp.Any):
        return not self.predicate(path, x)


class Everything:
    def __call__(self, path: Path, x: tp.Any):
        return True


class NonVariable:
    def __call__(self, path: Path, x: tp.Any):
        return not isinstance(x, nnx.Variable)


class Buffers:
    def __call__(self, path: Path, x: tp.Any):
        return isinstance(x, (jax.Array, np.ndarray))


buffers = Buffers()
