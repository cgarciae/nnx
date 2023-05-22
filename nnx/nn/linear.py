import dataclasses
import typing as tp

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from nnx import context

from nnx.module import Module
from nnx.nn import initializers
from nnx.dataclasses import dataclass, param
from nnx.nn import dtypes

Array = jax.Array
PRNGKey = tp.Any
Shape = tp.Tuple[int, ...]
Dtype = tp.Any  # this could be a real type?
PrecisionLike = tp.Union[
    None, str, lax.Precision, tp.Tuple[str, str], tp.Tuple[lax.Precision, lax.Precision]
]
ConvGeneralDilatedT = tp.Callable[..., Array]
PaddingLike = tp.Union[str, int, tp.Sequence[tp.Union[int, tp.Tuple[int, int]]]]
LaxPadding = tp.Union[str, tp.Sequence[tp.Tuple[int, int]]]
DotGeneralT = tp.Callable[..., Array]


default_kernel_init = initializers.lecun_normal()


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
    """ "Canonicalizes conv padding to a jax.lax supported format."""
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, tp.Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(
        f"Invalid padding format: {padding}, should be str, int,"
        f" or a sequence of len {rank} where each element is an"
        f" int or pair of ints."
    )


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class Linear(Module):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    # ref fields
    kernel: Array = param()
    bias: tp.Optional[Array] = param()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: tp.Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init,
        bias_init: tp.Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros(),
        dot_general: DotGeneralT = lax.dot_general,
        ctx: context.Context,
    ):
        kernel_key = ctx.make_rng("params")
        self.kernel = kernel_init(kernel_key, (in_features, out_features), param_dtype)
        if use_bias:
            bias_key = ctx.make_rng("params")
            self.bias = bias_init(bias_key, (out_features,), param_dtype)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dot_general = dot_general

    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.kernel
        bias = self.bias

        inputs, kernel, bias = dtypes.promote_dtype(
            inputs, kernel, bias, dtype=self.dtype
        )
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class Conv(Module):
    """Convolution Module wrapping `lax.conv_general_dilated[_local]`.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel. For 1D convolution,
        the kernel size can be passed as an integer. For all other cases, it must
        be a sequence of integers.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `'SAME'`, the string `'VALID'`, the string
        `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpeted as applying the same padding
        in all dims and passign a single int in a sequence causes the same padding
        to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    kernel: Array = param()
    bias: tp.Optional[Array] = param()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tp.Sequence[int],
        strides: tp.Union[None, int, tp.Sequence[int]] = 1,
        *,
        padding: PaddingLike = "SAME",
        input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
        kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        mask_fn: tp.Optional[tp.Callable[[Array], Array]] = None,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: tp.Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init,
        bias_init: tp.Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros(),
        conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
        ctx: context.Context,
    ):
        if isinstance(kernel_size, int):
            raise TypeError(
                "Expected Conv kernel_size to be a"
                " tuple/list of integers (eg.: [3, 3]) but got"
                f" {kernel_size}."
            )
        else:
            kernel_size = tuple(kernel_size)

        kernel_shape = kernel_size + (
            in_features // feature_group_count,
            out_features,
        )
        kernel_key = ctx.make_rng("params")
        self.kernel = kernel_init(kernel_key, kernel_shape, param_dtype)

        if use_bias:
            bias_shape = (out_features,)
            bias_key = ctx.make_rng("params")
            self.bias = bias_init(bias_key, bias_shape, param_dtype)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.input_dilation = input_dilation
        self.kernel_dilation = kernel_dilation
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.mask_fn = mask_fn
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.conv_general_dilated = conv_general_dilated

    def __call__(self, inputs: Array) -> Array:
        """Applies a (potentially unshared) convolution to the inputs.

        Args:
          inputs: input data with dimensions (*batch_dims, spatial_dims...,
            features). This is the channels-last convention, i.e. NHWC for a 2d
            convolution and NDHWC for a 3D convolution. Note: this is different from
            the input convention used by `lax.conv_general_dilated`, which puts the
            spatial dimensions last.
            Note: If the input has more than 1 batch dimension, all batch dimensions
            are flattened into a single dimension for the convolution and restored
            before returning.  In some cases directly vmap'ing the layer may yield
            better performance than this default flattening approach.  If the input
            lacks a batch dimension it will be added for the convolution and removed
            n return, an allowance made to enable writing single-example code.

        Returns:
          The convolved data.
        """

        assert isinstance(self.kernel_size, tuple)
        kernel_size = self.kernel_size

        def maybe_broadcast(
            x: tp.Optional[tp.Union[int, tp.Sequence[int]]]
        ) -> tp.Tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[num_batch_dimensions:]
            inputs = jnp.reshape(inputs, flat_input_shape)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == "CIRCULAR":
            kernel_size_dilated = [
                (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
            ]
            zero_pad: tp.List[tp.Tuple[int, int]] = [(0, 0)]
            pads = (
                zero_pad
                + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
                + [(0, 0)]
            )
            inputs = jnp.pad(inputs, pads, mode="wrap")
            padding_lax = "VALID"
        elif padding_lax == "CAUSAL":
            if len(kernel_size) != 1:
                raise ValueError(
                    "Causal padding is only implemented for 1D convolutions."
                )
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs = jnp.pad(inputs, pads)
            padding_lax = "VALID"

        dimension_numbers = _conv_dimension_numbers(inputs.shape)

        # One shared convolutional kernel for all pixels in the output.
        assert self.in_features % self.feature_group_count == 0

        kernel = self.kernel

        if self.mask_fn is not None:
            kernel = self.mask_fn(kernel)

        bias = self.bias

        inputs, kernel, bias = dtypes.promote_dtype(
            inputs, kernel, bias, dtype=self.dtype
        )

        y = self.conv_general_dilated(
            inputs,
            kernel,
            strides,
            padding_lax,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if self.use_bias:
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
        return y


default_embed_init = initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)


class Embed(Module):
    """Embedding Module.

    A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: same as embedding).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      embedding_init: embedding initializer.
    """

    embedding: Array = param()

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        embedding_init: tp.Callable[
            [PRNGKey, Shape, Dtype], Array
        ] = default_embed_init,
        ctx: context.Context,
    ):
        self.embedding = embedding_init(
            ctx.make_rng("params"),
            (num_embeddings, features),
            param_dtype,
        )

        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or self.embedding.dtype
        self.param_dtype = param_dtype
        self.embedding_init = embedding_init

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = dtypes.promote_dtype(
            self.embedding, dtype=self.dtype, inexact=False
        )
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.
        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        query, embedding = dtypes.promote_dtype(query, self.embedding, dtype=self.dtype)
        return jnp.dot(query, embedding.T)
