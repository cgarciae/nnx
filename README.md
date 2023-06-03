# nnx

_**N**eural **N**etworks for JA**X**_

`nnx` is a Neural Networks library for JAX that uses Pytree-based Modules and a novel 
a **Ref**erence system to enable:

* **Simplicity**: Provides an easy-to-understand mental model and implementation.
* **Shared refereces**: Supports **safe** mutable shared state thanks to its **Ref**erence system.
* **Leaf Metadata**: Enables semantic splitting a Module's state (similar to Flax collections) and adding [Axis Metadata](https://github.com/google/flax/blob/main/docs/flip/2434-general-metadata.md#flip-axis-metadata). 
* **Stateful transformations**: Seamless integration with JAX's native transformation capabilities.

NNX was designed to have the same capabilities as Flax with the simplicity of Equinox.

## Status

`nnx` is currently a proof of concept, aimed at exploring the design space of a lightweight module 
system for JAX based on references. It is not intended for production use.

## Installation

To get started with `nnx`, first install the package using pip:

```
pip install nnx
```
To get the latest version, install directly from GitHub:

```
pip install git+https://github.com/cgarciae/nnx
```

## Usage

```python
import nnx
import jax

class Linear(nnx.Module):

    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        key = ctx.make_rng("params")
        self.w = nnx.param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.param(jax.numpy.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w.value + self.b.value

ctx = nnx.Context(params=jax.random.PRNGKey(0))
model = Linear(din=12, dout=2, ctx=ctx)

@nnx.jit
def train_step(model, x, y):
    def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)
    
    # compute gradient
    grads: nnx.State = nnx.grad(loss_fn, wrt="params")(model)
    # SGD update
    model.update(
        jax.tree_map(lambda w, g: w - 0.1 * g, model.get("params"), grads)
    )

# yes... there's no return :)
train_step(model, x, y)
```

## Design

### Modules

`nnx` has a simple and intuitive design with Modules at its core. These modules are built using [simple_pytree](https://github.com/cgarciae/simple-pytree) Pytrees. To create a custom module, simply subclass `nnx.Module` and mark the reference fields with `nnx.ref` or `nnx.param`. These are descriptors that store a `Ref` instance in a separate attribute, making references transparent to the user.

Here's an example of a simple `Linear` module:

```python
import nnx
import jax

class Linear(nnx.Module):

    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        key = ctx.make_rng("params") # request an RNG key
        self.w = nnx.param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.param(jax.numpy.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w.value + self.b.value

ctx = nnx.Context(params=jax.random.PRNGKey(0))
model = Linear(din=12, dout=2, ctx=ctx)
```

`nnx` uses a `Context` object to propagate RNG and other forms of state throughout the model. `Context` provides a `make_rng` method that creates a new RNG key on demand for a given name (in this case, `"params"`).


#### GetItem and SetItem Syntactic Sugar

`Module` implements `__getitem__` and `__setitem__` to provide syntactic sugar for creating and updating `State`s. Although it may appear otherwise, `__setitem__` does not modify the Module's structure. Instead, it updates the values of the references, as demonstrated in this simplified implementation:

```python
class Module(nnx.Pytree):
    ...
    def __getitem__(self, collection: str) -> nnx.State:
        return nnx.get_partition(self, collection)

    def __setitem__(self, _: SetItemType, value: nnx.State):
        nnx.update_refs(self, value)
```

Sample usage might look like this:

```python
# SGD update
model.update(
    jax.tree_map(lambda w, g: w - 0.1 * g, model.get("params"), grads)
)
```

In this example, `model.get("params")` returns a `State` that contains all the references in the `params` collection. `grads` is a `State` with the same structure as `model.get("params")`, but with gradients instead of parameters. The statement `model.update(...)` updates the values of the references in `model` with the values of the new parameters from stochastic gradient descent (SGD) update rule.

### Transformations

Currently, NNX offers three types of transformations: stateful, filtered, and the partition API. As it is unclear which API is the best, all three will be maintained for now.

#### Stateful Transforms

Stateful transforms take a Pytree of references (e.g., a Module) as their first argument, track changes in the state of the references that occur within the transformation, and automatically propagate those changes to the input Pytree outside the transformation. In general, they have the following properties:

* They behave as stateful functions with respect to the first argument.
* They can operate on collections and RNG streams according to the transformation's semantics, exactly like Flax's transformations.
* They handle all relevant global state, such as `nnx.scope` and Refx's trace state.

Here's a diagram illustrating how stateful transformations work:

![stateful-transforms](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/stateful-transforms.png)

Currently, `nnx.jit` and `nnx.grad` are the only stateful transformations.

The following example demonstrates how a `train_step` function can be implemented using `nnx.jit` and `nnx.grad`:

```python
@nnx.jit
def train_step(model, x, y):

    def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)
    
    # compute gradient
    grads: nnx.State = nnx.grad(loss_fn, wrt="params")(model)
    # SGD update
    model.update(
        jax.tree_map(lambda w, g: w - 0.1 * g, model.get("params"), grads)
    )

# stateful update, no return needed
train_step(model, x, y)
```

The most interesting aspect of this design is that the code appears very imperative, as the state is automatically propagated in and out of the transformations. However, more consideration is needed to properly support `jit`'s `in_shardings` and `out_shardings` arguments.

Certainly! I've revised the text to improve clarity and readability. Here are the updated sections:

---
#### Filtered Transformations

Filtered transformations offer more flexibility, as they can take Pytrees of references in any of their arguments and return Pytrees of references. They simply `deref` and `merge` all their inputs and outputs to transfer the Pytrees across the transformation. In general, they have the following properties:

* They behave as pure functions.
* They don't handle any global state except for Refx's trace state.

![filtered-transforms](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/filtered-transforms.png)

Currently, `nnx.jit_filter` is the only filtered transformation.

Here's an example of how a `train_step` function can be implemented using `nnx.jit_filter` and `nnx.grad`:

```python
@nnx.jit_filter
def train_step(model, x, y):

    def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)

    # compute gradient
    grads: nnx.State = nnx.grad(loss_fn, wrt="params")(model)
    # SGD update
    model.update(
        jax.tree_map(lambda w, g: w - 0.1 * g, model.get("params"), grads)
    )
    
    return model

model = train_step(model, x, y)
```

Filtered transformations must output any state they want to propagate but have more flexibility in how they handle it. Adding support for `jit`'s `in_shardings` and `out_shardings` arguments is likely more straightforward than with stateful transformations.

#### Partition API

The partition API enables splitting a Module's state into sets of reference-less `State`s, this provides a general way of interacting with vanilla JAX transformations.


![partition-api](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/partition-api.png)

Here's an example of how a `train_step` function can be implemented using the partition API:

```python
states, moduledef = model.split()
params: nnx.State = states["params"]

@jax.jit
def train_step(params: nnx.State, x, y):

    def loss_fn(params: nnx.State):
        model: Linear = moduledef.merge(params)
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)

    # compute gradient
    grads: nnx.State = jax.grad(loss_fn)(params)
    # SGD update
    params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)
    
    return params

params = train_step(params, x, y)
```

The main benefit of the partition API is its compatibility with other JAX tools, as the training step can be written using regular JAX transformations. The main drawback is that it's more verbose.

### Case Studies

#### Shared State

In NNX, you can create modules that share state between them. This is useful when designing complex neural network architectures, as it allows you to reuse certain layers and reduce the number of learnable parameters.

Here's an example of creating a module with shared state:

```python
class Block(nnx.Module):
    def __init__(self, linear: nnx.Linear, *, ctx: nnx.Context):
        self.linear = linear
        self.bn = nnx.BatchNorm(2, ctx=ctx)

    def __call__(self, x, *, ctx: nnx.Context):
        x = self.linear(x)
        x = self.bn(x, ctx=ctx)
        x = nnx.relu(x)
        return x

class Model(nnx.Module):
    def __init__(self, *, ctx: nnx.Context):
        shared = nnx.Linear(2, 2, ctx=ctx)
        self.block1 = Block(shared, ctx=ctx)
        self.block2 = Block(shared, ctx=ctx)

    def __call__(self, x, *, ctx: nnx.Context):
        x = self.block1(x, ctx=ctx)
        x = self.block2(x, ctx=ctx)
        return x
```

In this example, the `Model` module contains two instances of the `Block` module. Each instance shares the same `nnx.Linear` module. To run the model, you can use the Context `flags` argument to set the `use_running_average` flag for all `BatchNorm` modules.

Here's an example of computing the loss for a `Model` instance:

```python
def loss_fn(model: Model, x: jax.Array, y: jax.Array):
    ctx = nnx.Context(flags=dict(use_running_average=True))
    y_pred = model(x, ctx=ctx)
    return jnp.mean((y - y_pred) ** 2)
```

It's important to note that the state for the shared `nnx.Linear` module will be kept in sync at all times on both `Block` instances, including during gradient updates.
