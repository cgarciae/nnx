# nnx

_**N**erual **N**etworks for JA**X**_

`nnx` is a lightweight module system for JAX that provides the same power as `flax` but with a simpler mental model and implementation. It is built on top of `refx`, which enables shared state, tractable mutability, and semantic partitioning. `nnx` also supports stateful transformations, allowing you to train your models efficiently.

## Status

`nnx` is currently a proof of concept, it is meant to explore the design space of a lightweight module system for JAX based on Refx.

## Installation

To get started with `nnx`, first install the package using pip:

```
pip install nnx
```

## Design

### Modules

* Modules are Pytrees
* `nnx.ref` and `nnx.param` mark ref fields, these are descriptor object (more info later)
* `make_rng` is similar to flax's `make_rng`, distributes RNG keys through global state
* the `init` and `apply` method lets you set the global state including RNG keys

```python
import nnx
import jax

# Modules are Pytrees
class Linear(nnx.Module):

    # mark parameter fields
    w: jax.Array = nnx.ref("params")
    b: jax.Array = nnx.param() # same as ref("params")

    def __init__(self, din: int, dout: int):
        key = self.make_rng("params") # request an RNG key
        self.w = jax.random.uniform(key, (din, dout))
        self.b = jax.numpy.zeros((dout,))

    def __call__(self, x):
        return x @ self.w + self.b

model = Linear.init(jax.random.PRNGKey(0))(din=12, dout=2)
```

#### RefField Descriptor
* `ref` and `params` create a `RefField` descriptor instances
* `RefField` descriptors inherit from `dataclasses.Field` in order to be compatible with `dataclasses` when needed
* `RefField` descriptors are descriptors that store a `Ref` instance in a separate `{name}__ref` attribute, this makes using references transparent to the user
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

#### GetItem and SetItem syntactic sugar
* `__getitem__` and `__setitem__` are syntactic sugar for `get_partition` and `update_refs`
* they don't actually update the Module structure despite the appearance, they just update the references values
```python
class Module(simple_pytree.Pytree):
    def __getitem__(self, collection: str) -> refx.Partition:
        derefed_module = deref(self)[0]
        return get_partition(derefed_module, collection)

    def __setitem__(self, collection: str, updates: refx.Partition):
        partition = get_partition(self, collection)
        update_refs(partition, updates)
```

### Transformations

* 3 types of transformations: stateful, filtered, and the partition API
* a final API would probably have a mix of the 3

#### Stateful Transformations

* stateful transformations take a pytree/Module of references as its first argument
* use `deref` and `reref` to move the pytree/Module across the transformation
* handle all relevant global state such as `nnx.scope` and Refx's trace state
* their main feature is that they always update the state of the references of the input pytree/Module
    
![stateful-transforms](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/stateful-transforms.png)

* `nnx.jit` and `nnx.grad` are stateful transformations

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

* probably the most "interesting" part of the design is that code looks very imperative
* state is automatically propagated in and out of the transformation

#### Filtered Transformations

* filtered transformations `deref` and `reref` all their inputs and outputs to move the pytree/Modules across the transformation
* they don't handle any global state, the must semantically behave as pure functions

![filtered-transforms](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/filtered-transforms.png)

* `nnx.jit_filter` is a filtered transformation, `nnx.grad` is a stateful transformation

```python
@nnx.jit_filter
def train_step(model, x, y):

    def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)

    # compute gradient
    grad = nnx.grad(loss_fn, wrt="params")(model)
    # sdg update                   |--------sdg-----------|
    model["params"] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)
    
    return model

model = train_step(model, x, y)
```

* filtered transformations must output any state that they want to propagate but have more flexibility in how they handle it


#### Partition API

* the partition API mimicks F

![partition-api](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/partition-api.png)

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
