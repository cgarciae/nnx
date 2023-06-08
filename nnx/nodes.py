import typing as tp

import jax
import numpy as np

node_types: tp.Tuple[type, ...] = ()


def register_node_type(node_type: type) -> None:
    global node_types
    node_types += (node_type,)


def is_node(obj: object) -> bool:
    return isinstance(obj, node_types)


# register nodes
register_node_type(jax.Array)
register_node_type(np.ndarray)
