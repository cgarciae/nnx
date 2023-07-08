__version__ = "0.0.0"


from .containers import (
    Container,
    ContainerMetadata,
    Node,
    Static,
    Variable,
    node,
    param,
    static,
    variable,
    with_metadata,
)
from .contextlib import Context, context
from .dataclasses import (
    dataclass,
    field,
    node_field,
    param_field,
    static_field,
    var_field,
)
from .errors import TraceContextError
from .helpers import Dict, Sequence, TrainState
from .module import Module, ModuleDef, Pure, PureModule
from .nn import initializers
from .nn.activations import (
    celu,
    elu,
    gelu,
    glu,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    logsumexp,
    normalize,
    one_hot,
    relu,
    relu6,
    selu,
    sigmoid,
    silu,
    soft_sign,
    softmax,
    softplus,
    standardize,
    swish,
    tanh,
)
from .nn.linear import Conv, Embed, Linear
from .nn.normalization import BatchNorm, LayerNorm
from .nn.stochastic import Dropout
from .nodes import is_node, register_node_type
from .partitioning import All, Not, buffers
from .pytreelib import Pytree, tree_node
from .spmd import (
    PARTITION_NAME,
    get_partition_spec,
    logical_axis_rules,
    logical_to_mesh,
    with_logical_constraint,
    with_logical_partitioning,
)
from .state import State
from .transforms import Remat, Scan, grad, jit, remat, scan
