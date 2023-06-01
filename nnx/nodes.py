import typing as tp

import jax
import numpy as np
from nnx.module import StateDef

from nnx.state import State

node_types: tp.Tuple[type, ...] = ()


def register_node_type(node_type: type) -> None:
    global node_types
    node_types += (node_type,)


def is_node_type(obj: object) -> bool:
    return isinstance(obj, node_types)


# register nodes
register_node_type(jax.Array)
register_node_type(np.ndarray)
register_node_type(State)
register_node_type(StateDef)
