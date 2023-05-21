from typing import Optional, Sequence

import jax.numpy as jnp
from jax import lax, random

from nnx.nn.module import Module
from nnx import rng_stream, utils
from nnx.dataclasses import dataclass, static_field


@dataclass
class Dropout(Module):
    """Create a dropout layer.

    Note: When using :meth:`Module.apply() <flax.linen.Module.apply>`, make sure
    to include an RNG seed named `'dropout'`. For example::

      model.apply({'params': params}, inputs=inputs, train=True, rngs={'dropout': dropout_rng})`

    Attributes:
      rate: the dropout probability.  (_not_ the keep rate!)
      broadcast_dims: dimensions that will share the same dropout mask
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
      rng_collection: the rng collection name to use when requesting an rng key.
    """

    rate: float
    broadcast_dims: Sequence[int] = static_field(default=())
    deterministic: Optional[bool] = static_field(default=None)
    rng_collection: str = static_field(default="dropout")

    def __call__(
        self,
        inputs,
        *,
        deterministic: Optional[bool] = None,
        rngs: Optional[rng_stream.Rngs],
    ):
        """Applies a random dropout mask to the input.

        Args:
          inputs: the inputs that should be randomly masked.
          deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
            masked, whereas if true, no mask is applied and the inputs are returned
            as is.

        Returns:
          The masked inputs reweighted to preserve mean.
        """
        deterministic = utils.first_from(
            deterministic,
            self.deterministic,
        )

        if (self.rate == 0.0) or deterministic:
            return inputs

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            return jnp.zeros_like(inputs)

        if rngs is None:
            raise ValueError(
                "Dropout needs to generate a random mask but no 'rngs' were provided."
            )

        keep_prob = 1.0 - self.rate
        rng = rngs.make_rng(self.rng_collection)
        broadcast_shape = list(inputs.shape)
        for dim in self.broadcast_dims:
            broadcast_shape[dim] = 1
        mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, inputs.shape)
        return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
