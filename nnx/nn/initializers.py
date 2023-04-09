import jax
from jax.nn.initializers import constant as constant
from jax.nn.initializers import delta_orthogonal as delta_orthogonal
from jax.nn.initializers import glorot_normal as glorot_normal
from jax.nn.initializers import glorot_uniform as glorot_uniform
from jax.nn.initializers import he_normal as he_normal
from jax.nn.initializers import he_uniform as he_uniform
from jax.nn.initializers import kaiming_normal as kaiming_normal
from jax.nn.initializers import kaiming_uniform as kaiming_uniform
from jax.nn.initializers import lecun_normal as lecun_normal
from jax.nn.initializers import lecun_uniform as lecun_uniform
from jax.nn.initializers import normal as normal
from jax.nn.initializers import orthogonal as orthogonal
from jax.nn.initializers import uniform as uniform
from jax.nn.initializers import variance_scaling as variance_scaling
from jax.nn.initializers import xavier_normal as xavier_normal
from jax.nn.initializers import xavier_uniform as xavier_uniform
from jax.nn.initializers import Initializer as Initializer


def zeros() -> Initializer:
    """Builds an initializer that returns a constant array full of zeros.

    >>> import jax, jax.numpy as jnp
    >>> from flax.linen.initializers import zeros_init
    >>> zeros_initializer = zeros_init()
    >>> zeros_initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)
    Array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    """
    return jax.nn.initializers.zeros


def ones() -> Initializer:
    """Builds an initializer that returns a constant array full of ones.

    >>> import jax, jax.numpy as jnp
    >>> from flax.linen.initializers import ones_init
    >>> ones_initializer = ones_init()
    >>> ones_initializer(jax.random.PRNGKey(42), (3, 2), jnp.float32)
    Array([[1., 1.],
           [1., 1.],
           [1., 1.]], dtype=float32)
    """
    return jax.nn.initializers.ones
