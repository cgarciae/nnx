# nnx

_**N**erual **N**etworks for JA**X**_

`nnx` is a lightweight module system for JAX that provides the same power as `flax` but with a simpler mental model and implementation. It is built on top of `refx`, which enables shared state, tractable mutability, and semantic partitioning. `nnx` also supports stateful transformations, allowing you to train your models efficiently.

## Status

`nnx` is currently a proof of concept and is meant to explore the design space of a lightweight module system for JAX based on Refx.

## Getting Started

To get started with `nnx`, first install the package using pip:

```
pip install nnx
```

Once you have installed `nnx`, you can define your modules as Pytrees. Here is an example of how to define a `Linear` module:

```python
import nnx
import jax

# Modules are Pytrees
class Linear(nnx.Module):

    # mark parameter fields
    w: jax.Array = nnx.param()
    b: jax.Array = nnx.param()

    def __init__(self, din: int, dout: int):
        key = self.make_rng("params") # request an RNG key
        self.w = jax.random.uniform(key, (din, dout))
        self.b = jax.numpy.zeros((dout,))

    def __call__(self, x):
        return x @ self.w + self.b
```

In this example, `Linear` is a Pytree with two fields: `w` and `b`. The `w` field is marked as a parameter using `nnx.ref`, and the `b` field is marked as a parameter using `nnx.param`.

To initialize a `Linear` module, you can use the `init` method:

```python
model = Linear.init(jax.random.PRNGKey(0))(12, 2)
```

This will create a `Linear` module with `din=12` and `dout=2`.

### Stateful Transformations

`nnx` supports stateful transformations, which allow you to train your models efficiently. Here is an example of how to define and use a stateful transformation with `nnx`:

```python
@nnx.jit
def train_step(model, x, y):

    def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)
    
    # compute gradient
    grad = nnx.grad(loss_fn, wrt="params")(model)
    # sdg update
    model["params"] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)

# stateful update, no return !!!
train_step(model, x, y)
```

In this example, `train_step` is a stateful transformation that takes a `model`, `x`, and `y` as inputs. The `loss_fn` function computes the loss of the model, and `nnx.grad` computes the gradient of the loss with respect to the parameters of the model. Finally, the `model` is updated using stochastic gradient descent.

### Shared State

In `nnx`, it's possible to create modules that share state between each other. This can be useful when designing complex neural network architectures, as it allows you to reuse certain layers and reduce the number of learnable parameters.

Here's an example of how to create a module with shared state:

```python
class Block(nnx.Module):
    def __init__(self, linear: nnx.Linear):
        self.linear = linear
        self.bn = nnx.BatchNorm(2)

    def __call__(self, x):
        return nnx.relu(self.bn(self.linear(x)))

class Model(nnx.Module):
    def __init__(self):
        shared = nnx.Linear(2, 2)
        self.block1 = Block(shared)
        self.block2 = Block(shared)

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
```

In this example, the `Model` module contains two instances of the `Block` module, each of which shares the same `nnx.Linear` module. To run the model you can use the `apply` method to set the `use_running_average` flag for all `BatchNorm` modules.

Here's an example of how to compute the loss for a `Model` instance:

```python
def loss_fn(model: Model, x: jax.Array, y: jax.Array):
    y_pred = model.apply(use_running_average=False)(x)
    return jnp.mean((y - y_pred) ** 2)
```

It's worth noting that the state for the shared `nnx.Linear` module will be kept in sync at all times on both `Block` instances, including during gradient updates.
