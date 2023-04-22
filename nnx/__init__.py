__version__ = "0.0.0"

from refx import Partition

from .fields import dataclass, field, param, ref, static_field
from .filters import jit_filter
from .nn import BatchNorm, Conv, Dropout, Linear, Module, ModuleDef
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
from .rng_stream import RngStream
from .scope_lib import (
    Scope,
    apply,
    current_scope,
    fork_scope,
    get_flag,
    init,
    make_rng,
    scope,
)
from .transforms import grad, jit
