from typing import Callable

import jax

import nnx


# lets pretend this function loads a pretrained model from a checkpoint
def load_backbone():
    return nnx.Linear(784, 128, ctx=nnx.context(0))


# create a simple linear classifier using a pretrained backbone
class Classifier(nnx.Module):
    def __init__(self, backbone: Callable[[jax.Array], jax.Array], *, ctx: nnx.Context):
        self.backbone = backbone
        self.head = nnx.Linear(128, 10, ctx=ctx)

    def __call__(self, x):
        x = self.backbone(x)
        x = nnx.relu(x)
        x = self.head(x)
        return x


# load the backbone
backbone = load_backbone()

# create the classifier using the pretrained backbone, here we are doing
# "parameter surgery", however, compared to Flax where you must manually
# construct the parameter structure, in NNX this is done automatically
model = Classifier(backbone, ctx=nnx.context(42))

# create a filter to select all the parameters that are not part of the
# backbone, i.e. the classifier parameters
is_trainable = nnx.All(lambda path, x: path[0] != "backbone", "params")

# partition the parameters into trainable and non-trainable parameters
(trainable_params, rest), moduledef = model.partition(is_trainable, ...)

print("trainable_params =", jax.tree_map(jax.numpy.shape, trainable_params))
print("rest = ", jax.tree_map(jax.numpy.shape, rest))
