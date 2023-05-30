__version__ = "0.0.0"


from .context import Context, RngStream
from .filters import jit_filter
from .module import Module, ModuleDef, DerefedMod
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
from .reference import (
    Partition,
    Variable,
    param,
    ref,
    RefMetadata,
    ref_metadata,
    with_partitioning,
)
from .transforms import grad, jit
from .module_helpers import Map, Seq
