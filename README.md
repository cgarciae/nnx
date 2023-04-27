# nnx

_**N**eural **N**etworks for JA**X**_

`nnx` is a lightweight module system for JAX designed to offer the same capabilities as `flax` but with a simpler mental model and implementation, inspired by `equinox`. It is built on top of `refx`, which enables shared state, tractable mutability, and semantic partitioning. `nnx` also supports stateful transformations, making it very simple to construct neural networks and other machine learning models, specially for begginers.

## Features

`nnx` offers a set of unique features that make it a user-friendly module system for JAX:

1. **Simplicity**: Designed with a straightforward mental model and implementation, `nnx` is easier to understand and use than some alternative JAX module systems.

2. **Shared state and tractable mutability**: `nnx` enables the creation of modules with shared state and controlled mutability, allowing for the reuse of layers and more efficient model design.

3. **Semantic partitioning**: Manage and manipulate different sections of your model more effectively with the ability to create partitions in your model.

4. **Stateful transformations**: Seamlessly integrate with JAX's native transformation capabilities, such as `jit`, `grad`, and more, using `nnx`'s support for stateful transformations.

## Status

`nnx` is currently a proof of concept, aimed at exploring the design space of a lightweight module system for JAX based on Refx. While it's still in the early stages of development, `nnx` has the potential to become a valuable tool for JAX users who seek a simpler alternative to existing module systems.

## Installation

To get started with `nnx`, first install the package using pip:

```
pip install nnx
```

## Usage

```python
import nnx
import jax

class Linear(nnx.Module):
    w: jax.Array = nnx.param()
    b: jax.Array = nnx.param()

    def __init__(self, din: int, dout: int):
        key = self.make_rng("params")
        self.w = jax.random.uniform(key, (din, dout))
        self.b = jax.numpy.zeros((dout,))

    def __call__(self, x):
        return x @ self.w + self.b

model = Linear.init(jax.random.PRNGKey(0))(din=12, dout=2)

@nnx.jit
def train_step(model, x, y):
    def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)
    
    grad = nnx.grad(loss_fn, wrt="params")(model)
    model["params"] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)

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
    w: jax.Array = nnx.ref("params")
    b: jax.Array = nnx.param() # shortcut for ref("params")

    def __init__(self, din: int, dout: int):
        key = self.make_rng("params") # request an RNG key
        self.w = jax.random.uniform(key, (din, dout))
        self.b = jax.numpy.zeros((dout,))

    def __call__(self, x):
        return x @ self.w + self.b
```

`nnx` offers the same `make_rng` API as Flax to distribute RNG keys where they are needed. It does this by storing RNG keys in a global state and carefully handling them with context managers. The `init` and `apply` methods allow you to set the state for the RNG keys and other flags. These methods are similar to Flax's `init` and `apply` but are designed to be more compatible with static analysis tools.

```python
# global state ==>  .....................
model = Linear.init(jax.random.PRNGKey(0))(din=12, dout=2)
#                    constructor args ==> ^^^^^^^^^^^^^^^^
```

If global state is not needed, you can simply use the constructor directly.

#### RefField Descriptor

`nnx.ref` and `nnx.param` are descriptors that create `RefField` instances. A `RefField` is a descriptor that stores a `Ref` instance in a separate `{attribute_name}__ref` attribute. It handles retrieving and setting the value of the reference automatically, so the user doesn't have to manipulate references directly. Additionally, `RefField` inherits from `dataclasses.Field` to ensure compatibility with `dataclasses` when needed.

Here's a simplified version of the `RefField` implementation:

```python
class RefField(dataclasses.Field):
    def __set_name__(self, cls, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        ref = getattr(obj, f"{self.name}__ref")
        return ref.value

    def __set__(self, obj, value):
        ref = getattr(obj, f"{self.name}__ref")
        ref.value = value
```

It's important to note that `Ref` instances are created during the first call to `__set__` if the companion `{name}__ref` attribute doesn't exist yet. This should only happen inside the `__init__` method, otherwise, the `Module` will raise an error, as `simple_pytree` Pytrees are frozen after initialization.

#### GetItem and SetItem Syntactic Sugar

`Module` implements `__getitem__` and `__setitem__` to provide syntactic sugar for creating and updating `Partition`s. Although it may appear otherwise, `__setitem__` does not modify the Module's structure. Instead, it updates the values of the references, as demonstrated in this simplified implementation:

```python
class Module(simple_pytree.Pytree):
    ...
    def __getitem__(self, collection: str) -> refx.Partition:
        derefed_module = refx.deref(self)[0]
        return nnx.get_partition(derefed_module, collection)

    def __setitem__(self, collection: str, updates: refx.Partition):
        partition = nnx.get_partition(self, collection)
        refx.update_refs(partition, updates)
```

Sample usage might look like this:

```python
model["params"] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)
```

In this example, `model["params"]` is a `Partition` that contains all the references in the `params` collection. `grad` is a `Partition` with the same structure as `model["params"]`, but with gradients instead of parameters. The expression `model["params"] = ...` updates the values of the references in `model["params"]` with the values of the stochastic gradient descent (SGD) update.

Of course! I've made some adjustments to the text to improve clarity and readability. Here's the revised version:

---
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
    grad = nnx.grad(loss_fn, wrt="params")(model)
    # SGD update
    model["params"] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)

# stateful update, no return needed
train_step(model, x, y)
```

The most interesting aspect of this design is that the code appears very imperative, as the state is automatically propagated in and out of the transformations. However, more consideration is needed to properly support `jit`'s `in_shardings` and `out_shardings` arguments.

Certainly! I've revised the text to improve clarity and readability. Here are the updated sections:

---
#### Filtered Transformations

Filtered transformations offer more flexibility, as they can take Pytrees of references in any of their arguments and return Pytrees of references. They simply `deref` and `reref` all their inputs and outputs to transfer the Pytrees across the transformation. In general, they have the following properties:

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
    grad = nnx.grad(loss_fn, wrt="params")(model)
    # SGD update
    model["params"] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)
    
    return model

model = train_step(model, x, y)
```

Filtered transformations must output any state they want to propagate but have more flexibility in how they handle it. Adding support for `jit`'s `in_shardings` and `out_shardings` arguments is likely more straightforward than with stateful transformations.

#### Partition API

The partition API mimics Flax's `variables` and `apply` API. It splits a Pytree of references into all its `Partition`s and creates a `ModuleDef` object that knows how to reconstruct the original Pytree from the `Partition`s. Since each `Partition` is a flat dictionary, this API works with regular JAX transformations.

Here's a diagram illustrating how the partition API works:

![partition-api](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/partition-api.png)

Here's an example of using the partition API:

```python
model: ModuleDef
partitions, model = model.partition()
params = partitions["params"]

@jax.jit
def train_step(params, x, y):

    def loss_fn(params):  #    |----merge----|
        y_pred, updates = model.apply([params])(x)
        return jax.numpy.mean((y_pred - y) ** 2)

    # compute gradient
    grad = jax.grad(loss_fn)(params)
    # SGD update
    params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grad)
    
    return params

params = train_step(params, x, y)
```

The main benefit of the partition API is its compatibility with other JAX tools, as the training step can be written using regular JAX transformations. The main drawback is that it's more verbose, and users must manually keep track of all the partitions. This overhead often makes `flax` and `haiku` a bit harder to learn than other frameworks like `pytorch` and `keras`.

### Case Studies

#### Shared State

In NNX, you can create modules that share state between them. This is useful when designing complex neural network architectures, as it allows you to reuse certain layers and reduce the number of learnable parameters.

Here's an example of creating a module with shared state:

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

In this example, the `Model` module contains two instances of the `Block` module. Each instance shares the same `nnx.Linear` module. To run the model, you can use the `apply` method to set the `use_running_average` flag for all `BatchNorm` modules.

Here's an example of computing the loss for a `Model` instance:

```python
def loss_fn(model: Model, x: jax.Array, y: jax.Array):
    y_pred = model.apply(use_running_average=False)(x)
    return jnp.mean((y - y_pred) ** 2)
```

It's important to note that the state for the shared `nnx.Linear` module will be kept in sync at all times on both `Block` instances, including during gradient updates.
