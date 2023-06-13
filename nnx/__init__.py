__version__ = "0.0.0"


from .containers import (
    Container,
    Node,
    Static,
    Variable,
    VarMetadata,
    node,
    param,
    static,
    var,
    var_metadata,
    with_partitioning,
)
from .contextlib import Context, RngStream, context
from .dataclasses import (
    dataclass,
    field,
    node_field,
    param_field,
    static_field,
    var_field,
)
from .errors import TraceContextError
from .helpers import Map, Sequence, TrainState
from .module import Module, ModuleDef, PureModule
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
from .nn.initializers import (
    Initializer,
    constant,
    delta_orthogonal,
    glorot_normal,
    glorot_uniform,
    he_normal,
    he_uniform,
    kaiming_normal,
    kaiming_uniform,
    lecun_normal,
    lecun_uniform,
    normal,
    ones,
    orthogonal,
    uniform,
    variance_scaling,
    xavier_normal,
    xavier_uniform,
    zeros,
)
from .nn.linear import Conv, Embed, Linear
from .nn.normalization import BatchNorm, LayerNorm
from .nn.stochastic import Dropout
from .nodes import is_node, register_node_type
from .partitioning import buffers
from .pytreelib import Pytree
from .state import State
from .transforms import grad, jit
