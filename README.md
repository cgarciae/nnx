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
import jax.numpy as jnp

class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        key = ctx.make_rng("params")
        self.w = nnx.param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.param(jax.numpy.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w + self.b

ctx = nnx.Context(params=jax.random.PRNGKey(0))
model = Linear(din=12, dout=2, ctx=ctx)
# forward pass
y = model(jnp.ones((8, 12)))
```
### Training

<details><summary>Stateful Transformations</summary>

```python
@nnx.jit
def train_step(model, x, y):
    def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)
    
    # compute gradient
    grads: nnx.State = nnx.grad(loss_fn, wrt="params")(model)
    # SGD update
    model.update_state(
        jax.tree_map(lambda w, g: w - 0.1 * g, model.filter("params"), grads)
    )

# yes... there's no return :)
train_step(model, x, y)
```

</details>

<details><summary>split API</summary>

```python
params, moduledef = model.split("params")

@jax.jit
def train_step(params, x, y):
    def loss_fn(params):
        y_pred, _updates = moduledef.apply(params)(x)
        return jax.numpy.mean((y_pred - y) ** 2)
    
    # compute gradient
    grads: nnx.State = jax.grad(loss_fn)(params)
    # SGD update
    params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)

    return params

params = train_step(params, x, y)
```

## Design

### Modules

NNX Modules are regular python classes that inherit from `nnx.Module`, they obey regular python semantics such as mutability and reference sharing, including reference cycles. They can contain 2 types of attributes: node attributes and static attributes. Node attributes include NNX `Variable`s (e.g. `nnx.param`), Numpy arrays, JAX arrays, and other Modules and other NNX types. All other types are treated as static attributes.

```python
class Foo(nnx.Module):
    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
        # node attributes
        self.variable = nnx.param(jnp.array(1))
        self.np_buffer = np.array(2)
        self.jax_buffer = jnp.array(3)
        self.submodule = nnx.Linear(din, dout, ctx=ctx)
        # static attributes
        self.din = din
        self.dout = dout

ctx = nnx.Context(jax.random.PRNGKey(0))
model = Foo(din=12, dout=2, ctx=ctx)
```
Regular python container types such as `list`, `tuple`, and `dict` are treated as static attributes, if similar functionality is needed, NNX provides the `Sequence` and `Map` Modules.

Since NNX Modules are not pytrees so they cannot be passed to JAX transformations. In order to interact with JAX, a Module must be split into the `State` and a `ModuleDef` objects. The `State` object is a flat dictionary-like structure that contains all the deduplicated node attributes, and the `ModuleDef` contains the static attributes and overall structural definition of the Module.

```python
state, moduledef = model.split()
print(state)
```
```
State({
    ('variable',): Variable(
        collection='params', value=Array(...)
    ),
    ('np_buffer',): Array(...),
    ('jax_buffer',): Array(...),
    ('submodule', 'kernel'): Variable(
        collection='params', value=Array(...)
    ),
    ('submodule', 'bias'): Variable(
        collection='params', value=Array(...),
    ),
})
```

`State` and `ModuleDef` are pytrees so they can be passed to JAX transformations. More over, `ModuleDef` provides 2 very important methods: `merge` and `apply`. The `merge` method can be used to create a new `Module` from a `State` object:

```python
model = moduledef.merge(state)
```
This can be use to e.g. recreate a module inside a JAX transformation. The `apply` provides a functional interface to the module, it can be used call any method on the Module and get both the output and the updated state:

```python
# run __call__
y, (state, moduledef) = moduledef.apply(state)(x)
# run some_method
y, (state, moduledef) = moduledef.apply(state).some_method(x)
```

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
    model.update_state(
        jax.tree_map(lambda w, g: w - 0.1 * g, model.filter("params"), grads)
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
    model.update_state(
        jax.tree_map(lambda w, g: w - 0.1 * g, model.filter("params"), grads)
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
