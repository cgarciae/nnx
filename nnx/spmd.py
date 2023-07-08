import collections
import contextlib
import dataclasses
import enum
import functools
import threading
import typing as tp

import jax
from jax.experimental import maps
from jax.sharding import Mesh, PartitionSpec

from nnx import containers
from nnx.nn import initializers
from nnx.state import State

# Real types and dummy aliases for documentation
LogicalRules = tp.Sequence[tuple[str, tp.Union[str, tuple[str, ...], None]]]
Array = tp.Any  # pylint: disable=invalid-name
ArrayPytree = tp.Any  # pylint: disable=invalid-name
LogicalPartitionSpec = tp.Any  # pylint: disable=invalid-name
LogicalPartitionSpecPytree = tp.Any  # pylint: disable=invalid-name
PartitionSpecPytree = tp.Any  # pylint: disable=invalid-name

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
PARTITION_NAME = "partition_name"
Sharding = tuple[tp.Optional[str], ...]


@tp.runtime_checkable
class HasSharding(tp.Protocol):
    sharding: tp.Optional[Sharding]


def add_axis(state: State, index: int, params: tp.Mapping[tp.Any, tp.Any]) -> State:
    axis_name = _get_partition_name(params)

    def _add_axis(x: tp.Any):
        if (
            isinstance(x, containers.Node)
            and isinstance(x, HasSharding)
            and x.sharding is not None
        ):
            sharding = list(x.sharding)
            while len(sharding) < index:
                sharding.append(None)
            sharding.insert(index, axis_name)
            return x.replace(sharding=tuple(sharding))
        return x

    return jax.tree_map(
        _add_axis, state, is_leaf=lambda x: isinstance(x, containers.Node)
    )


def remove_axis(state: State, index: int, params: tp.Mapping[tp.Any, tp.Any]) -> State:
    axis_name = _get_partition_name(params)

    def _remove_axis(x: tp.Any):
        if (
            isinstance(x, containers.Node)
            and isinstance(x, HasSharding)
            and x.sharding is not None
        ):
            sharding = list(x.sharding)
            assert sharding.pop(index) == axis_name
            return x.replace(sharding=tuple(sharding))
        return x

    return jax.tree_map(
        _remove_axis, state, is_leaf=lambda x: isinstance(x, containers.Node)
    )


def _get_partition_name(params: tp.Mapping[tp.Any, tp.Any]) -> str:
    if PARTITION_NAME not in params:
        raise ValueError(
            f'Trying to transform a Partitioned variable but "partition_name"'
            f" is not specified in metadata_params: {params}"
        )
    return params[PARTITION_NAME]


def get_partition_spec(tree: A) -> A:
    """Extracts a PartitionSpec tree from a PyTree containing ``Node`` values."""

    def f(x):
        if isinstance(x, containers.Node):
            if isinstance(x, HasSharding) and x.sharding:
                return PartitionSpec(*x.sharding)
            else:
                x = x.value

        # Unboxed arrays, which should be replicated across all devices
        if hasattr(x, "shape"):
            return PartitionSpec()
        else:
            return None

    return jax.tree_map(f, tree, is_leaf=lambda x: isinstance(x, containers.Node))


# Dynamic Axis Mapping Context
# ------------------------------------------------------------------------------


@dataclasses.dataclass
class _AxisRules(threading.local):
    """Dynamic logical axis to mesh axis binding context."""

    rules: LogicalRules = ()


# Global axis binding context.
_axis_rules = _AxisRules()


def set_logical_axis_rules(rules: LogicalRules):
    """Sets the global logical axis to mesh axis binding."""
    _axis_rules.rules = rules


def get_logical_axis_rules() -> LogicalRules:
    """Returns the global logical axis to mesh axis binding."""
    return _axis_rules.rules


@contextlib.contextmanager
def logical_axis_rules(rules: LogicalRules):
    """Context manager for setting the logical to mesh axis bindings."""
    old_rules = _axis_rules.rules
    try:
        _axis_rules.rules = rules
        yield
    finally:
        _axis_rules.rules = old_rules


class _UnassignedAxis:
    """Sentinel class for unassigned logical axis name."""

    def __repr__(self):
        return "UnassignedAxis"

    def __bool__(self):
        return False


_unassigned_axis = _UnassignedAxis()


def _mesh_assignment_free(new_assignment, existing_assignments):
    """Determines if a given mesh axis has already been assigned."""
    new = set(jax.tree_util.tree_leaves(new_assignment))
    existing = set(jax.tree_util.tree_leaves(existing_assignments))
    if existing.intersection(new):
        return False
    return True


def _logical_to_mesh_axes(
    array_dim_names: tp.Optional[tp.Sequence[tp.Optional[str]]],
    rules: tp.Optional[LogicalRules] = None,
) -> tp.Optional[list[tp.Union[_UnassignedAxis, None, str, tuple[str]]]]:
    """Same as logical_to_mesh_axes, but doesn't fill in _unassigned_axis."""
    if array_dim_names is None:
        return None
    if rules is None:
        rules = _axis_rules.rules
    axis_name_counts = collections.Counter(array_dim_names)
    dups = tuple(k for k, v in axis_name_counts.items() if v > 1 and k is not None)
    if dups:
        raise ValueError(
            f"Unsupported: Dimensions {dups} occur more than once in array names."
        )
    if not isinstance(rules, (tuple, list)):
        raise ValueError("Unknown axis rule specification type.")
    # We assign mesh axes using a priority based ruleset over logical axis names.
    result: list[tp.Union[_UnassignedAxis, None, str, tuple[str]]]
    result = [_unassigned_axis] * len(array_dim_names)
    for rule_model_name, rule_mesh_names in rules:
        if rule_model_name in array_dim_names:
            pos = array_dim_names.index(rule_model_name)
            if (
                _mesh_assignment_free(rule_mesh_names, result)
                and result[pos] == _unassigned_axis
            ):
                result[pos] = rule_mesh_names
    return result


def logical_to_mesh_axes(
    array_dim_names: tp.Optional[tp.Sequence[tp.Optional[str]]],
    rules: tp.Optional[LogicalRules] = None,
) -> tp.Optional[jax.sharding.PartitionSpec]:
    """Compute layout for an array.

    The rules are in order of precedence, and consist of pairs:
    (ArrayDimensionName, MeshDimensionName), meaning that the given array
    dimension (if present and unused) should be sharded across the given
    mesh dimension (if present and unused).

    A Layout of an Array is expressed as a tuple with one element for each
    dimension in the Array. The element is either None, or is the name of a
    mesh-dimension, meaning that this dimension of the array is sharded across
    this dimension of the mesh.

    For example, given an array with
      array_dim_names = ('batch', 'length', 'heads', 'features')
    and the layout rules are:
      rules = (('batch', 'X'),
               ('features', 'X'),
               ('heads', 'Y'),
               ('batch', 'Z'))

    then this function will return

      PartitionSpec('X', None, 'Y', None)

    Args:
      array_dim_names: tp.Tuple of array dimension names or None.
      rules: tp.Optional logical to mesh rules override.  Defaults to using the
        rules defined in the dynamic context set from the `axis_rules` function.

    Returns:
      PartitionSpec for the parameter.
    """
    result = _logical_to_mesh_axes(array_dim_names, rules)
    if result is None:
        return None
    # We default to None - ie unsharded along the dimension.
    result = [None if x is _unassigned_axis else x for x in result]
    return jax.sharding.PartitionSpec(*result)


def logical_to_mesh(tree: tp.Any, rules: tp.Optional[LogicalRules] = None) -> tp.Any:
    """Applies logical_to_mesh_axes to pytrees of logical PartitionSpecs."""
    return jax.tree_map(
        lambda x: logical_to_mesh_axes(x, rules),
        tree,
        is_leaf=lambda x: isinstance(x, jax.sharding.PartitionSpec),
    )


def logical_to_mesh_sharding(
    tree: tp.Any,
    mesh: jax.sharding.Mesh,
    rules: tp.Optional[LogicalRules] = None,
) -> tp.Any:
    """Convert pytrees of logical PartitionSpecs to shardings."""
    return jax.tree_map(
        lambda x: jax.sharding.NamedSharding(mesh, x),
        logical_to_mesh(tree, rules),
        is_leaf=lambda x: isinstance(x, jax.sharding.PartitionSpec),
    )


def _global_mesh_defined() -> bool:
    """Checks if global xmap/pjit mesh resource environment is defined."""
    maps_env = maps.thread_resources.env
    return (
        maps_env.physical_mesh.devices.shape != ()
    )  # pylint: disable=g-explicit-bool-comparison


class RulesFallback(enum.Enum):
    """How a sharding constraint should behave when no matching rule is found."""

    AXIS_IS_UNSHARDED = "axis_is_unsharded"
    RAISE_ERROR = "raise_error"
    NO_CONSTRAINT = "no_constraint"


def _with_sharding_constraint(
    x: Array,
    axis_resources: tp.Optional[jax.sharding.PartitionSpec],
    mesh: tp.Optional[jax.sharding.Mesh] = None,
):
    """Wrapper for pjit with_sharding_constraint, no-op on cpu or outside pjit."""
    # if jax.devices()[0].platform == "cpu" or (
    if not _global_mesh_defined() and mesh is None:
        return x
    else:
        if mesh is not None and axis_resources is not None:
            sharding = jax.sharding.NamedSharding(mesh, axis_resources)
            return jax.lax.with_sharding_constraint(x, sharding)
        return jax.lax.with_sharding_constraint(x, axis_resources)


def _with_sharding_constraint_one_fallback(
    axis_resources: LogicalPartitionSpec,
    x: Array,
    fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED,
    rules: tp.Optional[LogicalRules] = None,
    mesh: tp.Optional[jax.sharding.Mesh] = None,
):
    """Either imposes a sharding constraint or applies fallback."""
    mesh_axes = _logical_to_mesh_axes(axis_resources, rules)
    if mesh_axes is None:
        return _with_sharding_constraint(x, None, mesh=mesh)

    if fallback == RulesFallback.AXIS_IS_UNSHARDED:
        mesh_axes = [None if x is _unassigned_axis else x for x in mesh_axes]
    else:
        if any(x is _unassigned_axis for x in mesh_axes):
            if fallback == RulesFallback.RAISE_ERROR:
                raise ValueError(f"Axis names {axis_resources} did not match a rule")
            else:
                return x
    return _with_sharding_constraint(
        x, jax.sharding.PartitionSpec(*mesh_axes), mesh=mesh
    )


def _is_logical_spec(x):
    return x is None or (
        isinstance(x, tuple) and all(isinstance(e, str) or e is None for e in x)
    )


def with_logical_constraint(
    x: ArrayPytree,
    logical_axis_resources: LogicalPartitionSpecPytree,
    rules: tp.Optional[LogicalRules] = None,
    mesh: tp.Optional[jax.sharding.Mesh] = None,
    fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED,
):
    """Version of pjit's with_sharding_constraint that uses logical axis names."""
    # If no axis binding is set, this is a no-op.
    if rules is None:
        rules = _axis_rules.rules
    if not rules or logical_axis_resources is None:
        return x
    # Translate logical names to mesh assignments.
    return jax.tree_util.tree_map(
        functools.partial(
            _with_sharding_constraint_one_fallback,
            fallback=fallback,
            rules=rules,
            mesh=mesh,
        ),
        logical_axis_resources,
        x,
        is_leaf=_is_logical_spec,
    )


# Logical Partitioning Axis Metadata
# ------------------------------------------------------------------------------


@tp.runtime_checkable
class LogicallyPartitioned(tp.Protocol):
    unbox_fn: tp.Callable[[containers.Container[tp.Any]], tp.Any]
    sharding: Sharding
    mesh: tp.Optional[Mesh]
    rules: tp.Optional[LogicalRules]


def with_logical_partitioning(
    initializer: F,
    sharding: Sharding,
    mesh: tp.Optional[jax.sharding.Mesh] = None,
    rules: tp.Optional[LogicalRules] = None,
    **metadata: tp.Any,
) -> F:
    """Wraps a function's return value with LogicallyPartitioned.

    Example::

      kernel_init = with_logical_partitioning(
          nn.initializers.lecun_normal, (None, "data"))
      partitioned_dense = nn.Dense(features, kernel_init=kernel_init)

    Args:
      fn: The function to be wrapped. Typically this is an initializer.
      names: The logical axis passed to ``LogicallyPartitioned``.
      mesh: The mesh to use for the partitioning. If None, the global mesh
        resource is used if available.
      rules: tp.Optional logical to mesh rules use. If None, the global rules
        are used if available.
    Returns:
      A function wrapping ``fn`` that will return an instance of
      ``LogicallyPartitioned``.
    """

    def unbox_fn(node: containers.Node[tp.Any]) -> tp.Any:
        """Returns the wrapped value with the partitioning constraint applied."""
        if _global_mesh_defined() or (
            isinstance(node, LogicallyPartitioned) and node.mesh is not None
        ):
            return with_logical_constraint(
                node.value,
                get_partition_spec(node),
                rules=node.rules,
                mesh=node.mesh,
            )
        return node.value

    @functools.wraps(initializer)
    def wrapper(*args, **kwargs):
        y = initializer(*args, **kwargs)
        if _global_mesh_defined() or (mesh is not None):
            return with_logical_constraint(
                y,
                sharding,
                rules=rules,
                mesh=mesh,
            )
        return y

    return containers.with_metadata(
        tp.cast(F, wrapper),
        unbox_fn=unbox_fn,
        sharding=sharding,
        mesh=mesh,
        rules=rules,
        **metadata,
    )
