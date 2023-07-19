import typing as tp

import jax
import jax.tree_util as jtu

A = tp.TypeVar("A")
CollectionPredicate = tp.Callable[[str], bool]
Leaf = tp.Any
Leaves = tp.List[Leaf]
KeyPath = tp.Tuple[tp.Hashable, ...]
LeafPredicate = tp.Callable[[tp.Any], bool]


class Variable:
  __slots__ = ("_value", "_collection")

  def __init__(self, value: tp.Any, collection: str = "params"):
    ...

  @property
  def value(self) -> tp.Any:
    ...

  @property
  def collection(self) -> str:
    ...

  @classmethod
  def from_value(cls, value: tp.Any) -> "Variable":
    ...

  def copy(self) -> "Variable":
    ...

  def update(self, value: tp.Any) -> "Variable":
    ...

  def __repr__(self) -> str:
    return f"Variable({self.value}, collection={self.collection})"


def _flatten_variable_with_keys(variable: Variable):
  ...


def _flatten_variable(variable: Variable):
  ...


def _unflatten_variable(collection: str, nodes: tp.Tuple[tp.Any]):
  ...


jax.tree_util.register_pytree_with_keys(
    Variable,
    _flatten_variable_with_keys,
    _unflatten_variable,
    flatten_func=_flatten_variable,
)


class Nothing:

  def __repr__(self) -> str:
    ...


def _nothing_flatten(x):
  ...


def _nothing_unflatten(aux_data, children):
  ...


NOTHING = Nothing()

jtu.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


class StrPath(tp.Tuple[str, ...]):
  pass


class Partition(tp.Dict[tp.Tuple[str, ...], Leaf]):

  def __setitem__(self, key, value):
    raise TypeError("Partition is immutable")


def _partition_flatten_with_keys(
    x: Partition,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[StrPath, Leaf], ...], tp.Tuple[tp.Tuple[str, ...], ...]
]:
  ...


def _partition_unflatten(keys: tp.Tuple[StrPath, ...], leaves: tp.Tuple[Leaf, ...]):
  ...


jax.tree_util.register_pytree_with_keys(
    Partition, _partition_flatten_with_keys, _partition_unflatten
)


class PartitionDef(tp.Generic[A]):
  __slots__ = ("treedef",)

  def __init__(self, treedef: jtu.PyTreeDef):
    ...

  @property
  def treedef(self) -> jtu.PyTreeDef:
    ...

  def merge(self, *partitions: Partition) -> A:
    ...


def partitiondef_flatten(x: PartitionDef):
  ...


def statedef_unflatten(treedef, children):
  ...


jtu.register_pytree_node(PartitionDef, partitiondef_flatten, statedef_unflatten)


def tree_partition(
    pytree: A,
    *predicates: CollectionPredicate,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> tp.Tuple[tp.Tuple[Partition, ...], PartitionDef[A]]:
  ...


def get_partition(
    pytree,
    predicate: CollectionPredicate,
    is_leaf: tp.Optional[LeafPredicate] = None,
) -> Partition:
  ...


def merge_partitions(
    partitions: tp.Sequence[Partition], partitiondef: PartitionDef[A]
) -> A:
  ...
