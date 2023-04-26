# nnx

_**N**erual **N**etworks for JA**X**_

`nnx` is a lightweight module system for JAX that provides the same power as `flax` but with a simpler mental model and implementation closer to `equinox`. It is built on top of `refx`, which enables shared state, tractable mutability, and semantic partitioning. `nnx` also supports stateful transformations, allowing you to train your models efficiently.

## Status

`nnx` is currently a proof of concept, it is meant to explore the design space of a lightweight module system for JAX based on Refx.

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

<!-- * Modules are Pytrees
* `nnx.ref` and `nnx.param` mark ref fields, `params` is just a shorthand for `ref("params"), these are descriptor object (more info later)
* `make_rng` is similar to flax's `make_rng`, distributes RNG keys through global state
* the `init` and `apply` method lets you set the global state including RNG keys -->

NNX `Module`s are [simple_pytree](https://github.com/cgarciae/simple-pytree) Pytrees with a few additional features to make them more easier to use with `refx` references. A custom Module can be created simply by
subclassing `nnx.Module` and marking which fields are references with `nnx.ref` or `nnx.param`, as we will explain later, these are descriptors that store a `Ref` instance in a separate attribute and make using references transparent to the user.

Here is an example of a simple `Linear` module:

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
NNX offers the same `make_rng` API as Flax to distribute RNG keys where they are needed, it does this by storing RNG keys in a global state and carefully handling them with context managers. You can set the state for the RNG keys and other flags via the `init` and `apply` methods, which are similar to Flax's `init` and `apply` methods but designed to be friendlier with static analysis tools.

```python
# global state ==>  .....................
model = Linear.init(jax.random.PRNGKey(0))(din=12, dout=2)
#                    constructor args ==> ^^^^^^^^^^^^^^^^
```
If global state is not needed you can just use the constructor directly.

#### RefField Descriptor
<!-- * `ref` and `params` create a `RefField` descriptor instances
* `RefField` descriptors inherit from `dataclasses.Field` in order to be compatible with `dataclasses` when needed
* `RefField` descriptors are descriptors that store a `Ref` instance in a separate `{name}__ref` attribute, this makes using references transparent to the user -->

`nnx.ref` and `nnx.param` are descriptors that create `RefField` instances. `RefField` is a descriptor that stores a `Ref` instance in a separate `{attribute_name}__ref` attribute, and handle retrieving and setting the value of the reference automatically so that the user doesn't have to manipulate references directly. `RefField` inherits from `dataclasses.Field` in order to be compatible with `dataclasses` when needed.

Here is a simplified version of how `RefField` is implemented:

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

The only thing to note here is that `Ref`s are created during the first call to `__set__` if the companion `{name}__ref` attribute doesn't exist yet. This should only happen during `__init__` or else `Module` will raise an error as `simple_pytree` Pytrees are frozen after initialization.

#### GetItem and SetItem syntactic sugar
<!-- * `__getitem__` and `__setitem__` are syntactic sugar for `get_partition` and `update_refs`
* they don't actually update the Module structure despite the appearance, they just update the references values -->

`Module` implements `__getitem__` and `__setitem__` to provide syntactic sugar for creating and updating `Partition`s. Despite the appearance, `__setitem__` does not modify the Module's structure, it just updates the values of the references as can be seen in this simplified implementation:

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

Sample usage could look something like this:

```python
model["params"] = jax.tree_map(lambda w, g: w - 0.1 * g, model["params"], grad)
```

Here `model["params"]` is a `Partition` that contains all the references in the `params` collection, and `grad` is a `Partition` with the same structure as `model["params"]` but with gradients instead of parameters. `moduel["params"] = ...` updates the values of the references in `model["params"]` with the values of the `sdg` update.

### Transformations

<!-- * 3 types of transformations: stateful, filtered, and the partition API
* a final API would probably have a mix of the 3 -->

Currently NNX offers 3 types of transformations: stateful, filtered, and the partition API. At the moment its not clear which API is the best, the 3 will be kept for now.

#### Stateful Transforms

<!-- * stateful transformations take a pytree/Module of references as its first argument
* use `deref` and `reref` to move the pytree/Module across the transformation
* handle all relevant global state such as `nnx.scope` and Refx's trace state
* their main feature is that they always update the state of the references of the input pytree/Module -->

Stateful Transforms take a pytree of references (e.g. a Module) as their first argument, track changes in the state of the references that happen inside the transformation, and automatically propagate those changes to the input pytree outside the transformation. In general they have the following properties:

* they behave as stateful functions w.r.t. the first argument
* they can operate on collections and RNG streams according to the transformation's semantics, exactly like Flax's transformations
* they take care of handling all relevant global state such as `nnx.scope` and Refx's trace state
    
Here is a diagram of how stateful transformations work:

![stateful-transforms](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/stateful-transforms.png)

<!-- * `nnx.jit` and `nnx.grad` are stateful transformations -->

Currently `nnx.jit` and `nnx.grad` are the only stateful transformations.

Here is how a `train_step` function could be implemented using `nnx.jit` and `nnx.grad`:

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

<!-- * probably the most "interesting" part of the design is that code looks very imperative
* state is automatically propagated in and out of the transformation -->

The most interesting part of the design is that the code looks very imperative since the state is automatically propagated in and out of the transformations. However, more thought is needed to see how to correctly support `jit`'s `in_shardings` and `out_shardings` arguments.

#### Filtered Transformations

<!-- * filtered transformations `deref` and `reref` all their inputs and outputs to move the pytree/Modules across the transformation
* they don't handle any global state, the must semantically behave as pure functions -->

Filtered transformations are more flexible in that they can take pytrees of references in any of their arguments and also return pytrees of references. They just `deref` and `reref` all their inputs and outputs to move the pytrees across the transformation. In general they have the following properties:

* they behave as pure functions
* they don't handle any global state except for Refx's trace state

![filtered-transforms](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/filtered-transforms.png)

<!-- * `nnx.jit_filter` is a filtered transformation, `nnx.grad` is a stateful transformation -->

Currently `nnx.jit_filter` is the only filtered transformation.

Here is how a `train_step` function could be implemented using `nnx.jit_filter` and `nnx.grad`:

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

<!-- * filtered transformations must output any state that they want to propagate but have more flexibility in how they handle it -->

Filtered transformations must output any state that they want to propagate but have more flexibility in how they handle it. Adding support for `jit`'s `in_shardings` and `out_shardings` arguments is probably more straightforward than with stateful transformations.


#### Partition API

<!-- * the partition API mimicks Flax's `variables` + `apply` API
* it splits a pytree/Module into all its `Partition`s + a `ModuleDef` object
* each `Partition` is a flat dictionary so it works with regular JAX transformations -->

The partition API mimicks Flax's `variables` plus `apply` API. It splits a pytree of references into all its `Partition`s and creates a `ModuleDef` object that knows how to reconstruct the original pytree from the `Partition`s. Since each `Partition` is a flat dictionary, this API works with regular JAX transformations.
 
Here is a diagram of how the partition API works:

![partition-api](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/partition-api.png)

Here is an example of how to use the partition API:

```python
model: ModuleDef
partitions, model = model.partition()
params = partitions["params"]

@jax.jit
def train_step(params, x, y):

    def loss_fn(params): #      |----merge----|
        y_pred, updates = model.apply([params])(x)
        return jax.numpy.mean((y_pred - y) ** 2)

    # compute gradient
    grad = jax.grad(loss_fn)(params)
    # sdg update          |--------sdg-----------|
    params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grad)
    
    return params

params = train_step(params, x, y)
```

The main benefit of the partition API is that its more compatible with other JAX tools as the training step can be written using regular JAX transformations. The main drawback is that it's more verbose and users have to manually keep track of all the partitions, this overhead is often what makes `flax` and `haiku` a bit harder to learn than other frameworks like `pytorch` and `keras`.

### Case Studies

#### Shared State

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
