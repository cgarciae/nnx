import typing as tp

import jax

from nnx import containers
from nnx.state import State

PARTITION_NAME = "partition_name"


def add_axis(state: State, index: int, params: tp.Dict[tp.Any, tp.Any]) -> State:
    axis_name = _get_partition_name(params)

    def _add_axis(x: tp.Any):
        if isinstance(x, containers.Variable) and x.sharding is not None:
            sharding = list(x.sharding)
            while len(sharding) < index:
                sharding.append(None)
            sharding.insert(index, axis_name)
            return x.replace_metadata(sharding=tuple(sharding))
        return x

    return jax.tree_map(_add_axis, state)


def remove_axis(state: State, index: int, params: tp.Dict[tp.Any, tp.Any]) -> State:
    axis_name = _get_partition_name(params)

    def _remove_axis(x: tp.Any):
        if isinstance(x, containers.Variable) and x.sharding is not None:
            sharding = list(x.sharding)
            assert sharding.pop(index) == axis_name
            return x.replace_metadata(sharding=tuple(sharding))
        return x

    return jax.tree_map(_remove_axis, state)


def _get_partition_name(params: tp.Dict[tp.Any, tp.Any]) -> str:
    if PARTITION_NAME not in params:
        raise ValueError(
            f'Trying to transform a Partitioned variable but "partition_name"'
            f" is not specified in metadata_params: {params}"
        )
    return params[PARTITION_NAME]
