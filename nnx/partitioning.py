import builtins
import dataclasses
import functools
import typing as tp
from math import e

import jax
import jax.tree_util as jtu
import numpy as np

import nnx

if tp.TYPE_CHECKING:
    ellipsis = builtins.ellipsis
else:
    ellipsis = tp.Any

Predicate = tp.Callable[[tp.Tuple[str, ...], tp.Any], bool]
CollectionFilter = tp.Union[
    str,
    tp.Sequence[str],
    Predicate,
    ellipsis,
]


def to_predicate(collection_filter: CollectionFilter) -> Predicate:
    if isinstance(collection_filter, str):
        return Is(collection_filter)
    elif collection_filter is Ellipsis:
        return Everything()
    elif callable(collection_filter):
        return collection_filter
    elif isinstance(collection_filter, tp.Sequence):
        return Any(collection_filter)
    else:
        raise TypeError(f"Invalid collection filter: {collection_filter}")


@dataclasses.dataclass
class Is:
    collection: str

    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        if isinstance(x, nnx.Variable):
            return x.collection == self.collection
        return False


class Any:
    def __init__(self, collection_filters: tp.Sequence[CollectionFilter]):
        self.predicates = tuple(
            to_predicate(collection_filter) for collection_filter in collection_filters
        )

    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        return any(predicate(path, x) for predicate in self.predicates)


class Not:
    def __init__(self, collection_filter: CollectionFilter):
        self.predicate = to_predicate(collection_filter)

    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        return not self.predicate(path, x)


class Everything:
    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        return True


class NonConstant:
    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        return not isinstance(x, nnx.Variable)


class Buffers:
    def __call__(self, path: tp.Tuple[str, ...], x: tp.Any):
        return isinstance(x, (jax.Array, np.ndarray))


buffers = NonConstant()
buffers = Buffers()
