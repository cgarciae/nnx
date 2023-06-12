# %%
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets import load_dataset

import nnx

np.random.seed(42)
latent_size = 32
image_shape: tp.Sequence[int] = (28, 28)
steps_per_epoch: int = 200
batch_size: int = 64
epochs: int = 20


dataset = load_dataset("mnist")
X_train = np.array(np.stack(dataset["train"]["image"]), dtype=np.uint8)
X_test = np.array(np.stack(dataset["test"]["image"]), dtype=np.uint8)
# Now binarize data
X_train = (X_train > 0).astype(jnp.float32)
X_test = (X_test > 0).astype(jnp.float32)

print("X_train:", X_train.shape, X_train.dtype)
print("X_test:", X_test.shape, X_test.dtype)


# %%
class Encoder(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, ctx: nnx.Context):
        self.linear1 = nnx.Linear(din, dmid, ctx=ctx)
        self.linear_mean = nnx.Linear(dmid, dout, ctx=ctx)
        self.linear_std = nnx.Linear(dmid, dout, ctx=ctx)

    def __call__(self, x: jax.Array, *, ctx: nnx.Context) -> jax.Array:
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.linear1(x)
        x = jax.nn.relu(x)

        mean = self.linear_mean(x)
        std = jnp.exp(self.linear_std(x))

        self.kl_loss = nnx.var(
            "losses",
            jnp.mean(
                0.5 * jnp.mean(-jnp.log(std**2) - 1.0 + std**2 + mean**2, axis=-1)
            ),
        )
        key = ctx.make_rng("noise")
        z = mean + std * jax.random.normal(key, mean.shape)
        return z


class Decoder(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, ctx: nnx.Context):
        self.linear1 = nnx.Linear(din, dmid, ctx=ctx)
        self.linear2 = nnx.Linear(dmid, dout, ctx=ctx)

    def __call__(self, z: jax.Array) -> jax.Array:
        z = self.linear1(z)
        z = jax.nn.relu(z)
        logits = self.linear2(z)
        return logits


class VAE(nnx.Module):
    def __init__(
        self,
        din: int,
        hidden_size: int,
        latent_size: int,
        output_shape: tp.Sequence[int],
        *,
        ctx: nnx.Context,
    ):
        self.output_shape = output_shape
        self.encoder = Encoder(din, hidden_size, latent_size, ctx=ctx)
        self.decoder = Decoder(
            latent_size, hidden_size, int(np.prod(output_shape)), ctx=ctx
        )

    def __call__(self, x: jax.Array, *, ctx: nnx.Context) -> jax.Array:
        z = self.encoder(x, ctx=ctx)
        logits = self.decoder(z)
        logits = jnp.reshape(logits, (-1, *self.output_shape))
        return logits

    def generate(self, z):
        logits = self.decoder(z)
        logits = jnp.reshape(logits, (-1, *self.output_shape))
        return nnx.sigmoid(logits)


params, moduledef = VAE(
    din=int(np.prod(image_shape)),
    hidden_size=256,
    latent_size=latent_size,
    output_shape=image_shape,
    ctx=nnx.context(0),
).partition("params")

state = nnx.TrainState(
    apply_fn=moduledef.apply,
    params=params,
    tx=optax.adam(1e-3),
)


# %%
@jax.jit
def train_step(state: nnx.TrainState[VAE], x: jax.Array, key: jax.Array):
    def loss_fn(params: nnx.State):
        ctx = nnx.context(noise=jax.random.fold_in(key, state.step))
        logits, (updates, _) = state.apply_fn(params)(x, ctx=ctx)

        losses = updates.filter("losses")
        kl_loss = sum(jax.tree_util.tree_leaves(losses), 0.0)
        reconstruction_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, x))

        loss = reconstruction_loss + 0.1 * kl_loss
        return loss, loss

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, loss = grad_fn(state.params)
    state.apply_gradients(grads=grads)

    return state, loss


@partial(jax.jit, donate_argnums=(0,))
def forward(state: nnx.TrainState[VAE], x: jax.Array, key: jax.Array) -> jax.Array:
    ctx = nnx.context(noise=key)
    y_pred = state.apply_fn(state.params)(x, ctx=ctx)[0]
    return jax.nn.sigmoid(y_pred)


@jax.jit
def sample(state: nnx.TrainState[VAE], z: jax.Array) -> jax.Array:
    return state.apply_fn(state.params).generate(z)[0]


# %%
key = jax.random.PRNGKey(0)

for epoch in range(epochs):
    losses = []
    for step in range(steps_per_epoch):
        idxs = np.random.randint(0, len(X_train), size=(batch_size,))
        x_batch = X_train[idxs]

        state, loss = train_step(state, x_batch, key)
        losses.append(np.asarray(loss))

    print(f"Epoch {epoch} loss: {np.mean(losses)}")

exit()
# %%
# get random samples
idxs = np.random.randint(0, len(X_test), size=(5,))
x_sample = X_test[idxs]

# get predictions
y_pred = forward(state, x_sample, key)

# plot reconstruction
figure = plt.figure(figsize=(3 * 5, 3 * 2))
plt.title("Reconstruction Samples")
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_sample[i], cmap="gray")
    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(y_pred[i], cmap="gray")
    # # tbwriter.add_figure("VAE Example", figure, epochs)

plt.show()

# %%
# plot generative samples
z_samples = np.random.normal(scale=1.5, size=(12, latent_size))
samples = sample(state, z_samples)

figure = plt.figure(figsize=(3 * 5, 3 * 2))
plt.title("Generative Samples")
for i in range(5):
    plt.subplot(2, 5, 2 * i + 1)
    plt.imshow(samples[i], cmap="gray")
    plt.subplot(2, 5, 2 * i + 2)
    plt.imshow(samples[i + 1], cmap="gray")

plt.show()

# %%
