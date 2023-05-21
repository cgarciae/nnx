__version__ = "0.0.0"


from .dataclasses import dataclass, field, param, ref, static_field
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
from .partitioning import (
    collection_partition as partition,
)
from .partitioning import (
    get_partition,
    tree_partition,
)
from .partitioning import (
    merge_partitions as merge,
)
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
from .rng_stream import Rngs, RngStream
from .transforms import grad, jit

from .filters import jit_filter
