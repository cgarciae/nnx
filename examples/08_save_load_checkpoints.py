from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax

import nnx


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, ctx: nnx.Context):
        self.dense1 = nnx.Linear(din, dmid, ctx=ctx)
        self.dense2 = nnx.Linear(dmid, dout, ctx=ctx)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        return x


def create_model(seed: int):
    return MLP(10, 20, 30, ctx=nnx.context(seed))


def create_and_save(seed: int, path: str):
    model = create_model(seed)
    state = model.get_state()
    # Save the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(f"{path}/state", state)


def load_model(path: str) -> MLP:
    # create that model with abstract shapes
    state, moduledef = jax.eval_shape(lambda: create_model(0).partition())
    # Load the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    state = checkpointer.restore(f"{path}/state", item=state)
    # Merge the parameters into the model
    model = moduledef.merge(state)
    return model


with TemporaryDirectory() as tmpdir:
    # create a checkpoint
    create_and_save(42, tmpdir)
    # load model from checkpoint
    model = load_model(tmpdir)
    # run the model
    y = model(jnp.ones((1, 10)))
    print(model)
    print(y)
