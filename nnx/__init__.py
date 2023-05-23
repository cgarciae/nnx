__version__ = "0.0.0"


from .context import Context, RngStream
from .dataclasses import dataclass, field, node_field, param, ref, static_field
from .filters import jit_filter
from .module import Module, ModuleDef, Bounded
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
from .partitioning import collection_partition as partition
from .partitioning import get_partition
from .partitioning import merge_partitions as merge
from .partitioning import tree_partition
from .pytree import Pytree, PytreeMeta
from .ref_field import RefField
from .reference import (
    NOTHING,
    Dag,
    DagDef,
    Deref,
    Index,
    Partition,
    Ref,
    Referential,
    Value,
    clone,
    deref,
    reref,
    update_refs,
)
from .transforms import grad, jit
