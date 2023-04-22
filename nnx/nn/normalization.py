import typing as tp

import jax
import jax.numpy as jnp
from jax import lax

from nnx.nn.module import Module
from nnx.nn import initializers, dtypes
from nnx import fields, utils

PRNGKey = jax.Array
Array = jax.Array
Shape = tp.Tuple[int, ...]
Dtype = tp.Any  # this could be a real type?

Axes = tp.Union[int, tp.Any]


def _canonicalize_axes(rank: int, axes: Axes) -> tp.Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, tp.Iterable):
        axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
        return lax.square(lax.real(x)) + lax.square(lax.imag(x))
    else:
        return lax.square(x)


def _compute_stats(
    x: Array,
    axes: tp.Optional[Axes],
    dtype: tp.Optional[Dtype],
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_mean: bool = True,
):
    """Computes mean and variance statistics.

    This implementation takes care of a few important details:
    - Computes in float32 precision for stability in half precision training.
    - mean and variance are computable in a single XLA fusion,
      by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
    - Clips negative variances to zero which can happen due to
      roundoff errors. This avoids downstream NaNs.
    - Supports averaging across a parallel axis and subgroups of a parallel axis
      with a single `lax.pmean` call to avoid latency.

    Arguments:
      x: Input array.
      axes: The axes in ``x`` to compute mean and variance statistics for.
      dtype: tp.Optional dtype specifying the minimal precision. Statistics
        are always at least float32 for stability (default: dtype of x).
      axis_name: tp.Optional name for the pmapped axis to compute mean over.
      axis_index_groups: tp.Optional axis indices.
      use_mean: If true, calculate the mean from the input and use it when
        computing the variance. If false, set the mean to zero and compute
        the variance without subtracting the mean.

    Returns:
      A pair ``(mean, var)``.
    """
    if dtype is None:
        dtype = jnp.result_type(x)
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jnp.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)

    mean2 = jnp.mean(_abs_sq(x), axes)
    if use_mean:
        mean = jnp.mean(x, axes)
    else:
        mean = jnp.zeros(mean2.shape, dtype=dtype)

    if axis_name is not None:
        concatenated_mean = jnp.concatenate([mean, mean2])
        mean, mean2 = jnp.split(
            lax.pmean(
                concatenated_mean,
                axis_name=axis_name,
                axis_index_groups=axis_index_groups,
            ),
            2,
        )
    # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
    # to floating point round-off errors.
    var = jnp.maximum(0.0, mean2 - _abs_sq(mean))
    return mean, var


def _normalize(
    x: Array,
    mean: Array,
    var: Array,
    scale: tp.Optional[Array],
    bias: tp.Optional[Array],
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Dtype,
    epsilon: float,
):
    """ "Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

    Arguments:
      x: The input.
      mean: Mean to use for normalization.
      var: Variance to use for normalization.
      reduction_axes: The axes in ``x`` to reduce.
      feature_axes: Axes containing features. A separate bias and scale is learned
        for each specified feature.
      dtype: The dtype of the result (default: infer from input and params).
      epsilon: Normalization epsilon.

    Returns:
      The normalized input.
    """
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    args = [x]
    if scale is not None:
        scale = scale.reshape(feature_shape)
        mul *= scale
        args.append(scale)
    y *= mul
    if bias is not None:
        bias = bias.reshape(feature_shape)
        y += bias
        args.append(bias)
    dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


@fields.dataclass
class BatchNorm(Module):
    """BatchNorm Module.

    Usage Note:
    If we define a model with BatchNorm, for example::

      BN = nn.BatchNorm(use_running_average=False, momentum=0.9, epsilon=1e-5,
                        dtype=jnp.float32)

    The initialized variables dict will contain in addition to a 'params'
    collection a separate 'batch_stats' collection that will contain all the
    running statistics for all the BatchNorm layers in a model::

      vars_initialized = BN.init(key, x)  # {'params': ..., 'batch_stats': ...}

    We then update the batch_stats during training by specifying that the
    `batch_stats` collection is mutable in the `apply` method for our module.::

      vars_in = {'params': params, 'batch_stats': old_batch_stats}
      y, mutated_vars = BN.apply(vars_in, x, mutable=['batch_stats'])
      new_batch_stats = mutated_vars['batch_stats']

    During eval we would define BN with `use_running_average=True` and use the
    batch_stats collection from training to set the statistics.  In this case
    we are not mutating the batch statistics collection, and needn't mark it
    mutable::

      vars_in = {'params': params, 'batch_stats': training_batch_stats}
      y = BN.apply(vars_in, x)

    Attributes:
      use_running_average: if True, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of
        the batch statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma).
        When the next layer is linear (also e.g. nn.relu), this can be disabled
        since the scaling will be done by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over
        the examples on the first two and last two devices. See `jax.lax.psum`
        for more details.
    """

    num_features: int = fields.static_field()
    use_running_average: tp.Optional[bool] = fields.static_field(default=None)
    axis: int = fields.static_field(default=-1)
    momentum: float = fields.static_field(default=0.99)
    epsilon: float = fields.static_field(default=1e-5)
    dtype: tp.Optional[Dtype] = fields.static_field(default=None)
    param_dtype: Dtype = fields.static_field(default=jnp.float32)
    use_bias: bool = fields.static_field(default=True)
    use_scale: bool = fields.static_field(default=True)
    bias_init: tp.Callable[[PRNGKey, Shape, Dtype], Array] = fields.static_field(
        default=initializers.zeros()
    )
    scale_init: tp.Callable[[PRNGKey, Shape, Dtype], Array] = fields.static_field(
        default=initializers.ones()
    )
    axis_name: tp.Optional[str] = fields.static_field(default=None)
    axis_index_groups: tp.Any = fields.static_field(default=None)
    # refs
    mean: Array = fields.ref("batch_stats", init=False)
    var: Array = fields.ref("batch_stats", init=False)
    scale: tp.Optional[Array] = fields.param(init=False)
    bias: tp.Optional[Array] = fields.param(init=False)

    def __post_init__(self):
        feature_shape = (self.num_features,)
        self.mean = jnp.zeros(feature_shape, jnp.float32)
        self.var = jnp.ones(feature_shape, jnp.float32)

        if self.use_scale:
            key = self.make_rng("params")
            self.scale = self.scale_init(key, feature_shape, self.param_dtype)
        else:
            self.scale = None

        if self.use_bias:
            key = self.make_rng("params")
            self.bias = self.bias_init(key, feature_shape, self.param_dtype)
        else:
            self.bias = None

    def __call__(self, x, use_running_average: tp.Optional[bool] = None):
        """Normalizes the input using batch statistics.

        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats
            will be used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = utils.first_from(
            use_running_average,
            self.use_running_average,
            self.get_flag("use_running_average", None),
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

        if use_running_average:
            mean, var = self.mean, self.var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups,
            )

            self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
            self.var = self.momentum * self.var + (1 - self.momentum) * var

        return _normalize(
            x,
            mean,
            var,
            self.scale,
            self.bias,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.epsilon,
        )
