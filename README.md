[![codecov](https://codecov.io/gh/cgarciae/nnx/branch/main/graph/badge.svg?token=VqJjL474Z7)](https://codecov.io/gh/cgarciae/nnx)

# NNX

_**N**eural **N**etworks for JA**X**_

NNX is a Neural Networks library for JAX that provides a simple module system that respects regular python semantics. Its designed to be as powerful as [Flax](https://flax.readthedocs.io/en/latest/) but with a highly simplified pythonic API reminiscent of [PyTorch](https://pytorch.org/)

* **Vanilla Python Semantics**: Modules are normal python classes that respect regular python semantics such as mutability and reference sharing. No mandatory dataclass behavior.
* **Safety**: NNX is designed to with safety in mind, it includes mechanisms to prevent tracer leakage, avoid stale RNGs, and proper state propagation.
* **Semantic Partitioning**: NNX enables you mark attributes as members of specific collections such as `params`, `batch_stats`, etc, so that each collection can be processed independently when needed.
* **Lifted transforms**: NNX provides a set of Module-aware transforms that take care of handling the Module's state and provide APIs to process each collection differently by the underlying JAX transform.

## Installation

To get started with `nnx`, first install the package using pip:

```
pip install nnx
```
To get the latest version, install directly from GitHub:

```
pip install git+https://github.com/cgarciae/nnx
```

## Basic Usage

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

<details><summary>Stateful Transforms</summary>

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

<details><summary>Functional API</summary>

```python
params, moduledef = model.partition("params")

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

</details>

## Design

### Modules

NNX Modules are regular python classes that inherit from `nnx.Module`, they obey regular python semantics such as mutability and reference sharing, including reference cycles. They can contain 2 types of attributes: node attributes and static attributes. Node attributes include NNX `Variable`s (e.g. `nnx.param`), Numpy arrays, JAX arrays, submodules Modules, and other NNX types. All other types are treated as static attributes.

```python
class Foo(nnx.Module):
    def __init__(self, ctx: nnx.Context):
        # node attributes
        self.variable = nnx.param(jnp.array(1))
        self.np_buffer = np.array(2)
        self.jax_buffer = jnp.array(3)
        self.node = nnx.node([4, 5])
        self.submodule = nnx.Linear(2, 4, ctx=ctx)
        # static attributes
        self.int = 1
        self.float = 2.0
        self.str = "hello"
        self.list = [1, 2, 3]

ctx = nnx.Context(jax.random.PRNGKey(0))
model = Foo(din=12, dout=2, ctx=ctx)
```
As shown above, python container types such as `list`, `tuple`, and `dict` are treated as static attributes, if similar functionality is needed, NNX provides the `Sequence` and `Map` Modules.

### Functional API

NNX Modules are not pytrees so they cannot be passed to JAX transformations. In order to interact with JAX, a Module must be partitioned into a `State` and `ModuleDef` objects. The `State` object is a flat dictionary-like pytree structure that contains all the deduplicated node attributes, and the `ModuleDef` contains the static attributes and structural information needed to reconstruct the Module.

```python
state, moduledef = model.partition()
```
```
State({
  ('jax_buffer',): Array(3),
  ('node',): Node(value=[4, 5]),
  ('np_buffer',): array(2),
  ('submodule', 'bias'): Variable(collection='params', value=Array(...)),
  ('submodule', 'kernel'): Variable(collection='params', value=Array(...)),
  ('variable',): Variable(collection='params', value=Array(1))
})
```

`State` and `ModuleDef` are pytrees so they can be passed to JAX transformations. More over, `ModuleDef` provides 2 very important methods: `merge` and `apply`. The `merge` method can be used to create a new `Module` from a `State` object:

```python
model = moduledef.merge(state)
```
This can be use to e.g. recreate a module inside a JAX transformation. The `apply` provides a functional interface to the module, it can be used call any method or submodule and get the output and the updated state:

```python
# run __call__
y, (state, moduledef) = moduledef.apply(state)(x)
# run some_method
y, (state, moduledef) = moduledef.apply(state).some_method(x)
# run submodule
y, (state, moduledef) = moduledef.apply(state).submodule(x)
```

`apply` can call any nested method or submodule as long as it can be accessed via the `.` or `[]` operators.

### Partitioning State
`nnx.var` lets you create `Variable` attributes with `collection` metadata, this metadata can be used to partition the state into multiple substates. `nnx.param` is a special case of `nnx.var` where `collection="params"`.

```python
class Foo(nnx.Module):
    def __init__(self):
        self.a = nnx.param(1.0)
        self.b = nnx.var("batch_stats", 2.0)
        self.c = jax.numpy.array(3.0)

    def __call__(self, x):
        return x * self.a + self.b

model = Foo()
```
Collection names can be passed as filters to the `partition` method to split the state into mutually exclusive substates containing only the node attributes with the corresponding collection name:

```python
(params, batch_stats, rest), moduledef = model.partition("params", "batch_stats", ...)
```
`partition` will make sure all nodes are match by atleast one filter, else it will raise an error. If you have non-`Variable` nodes like `nnx.node`, `jax.Array`, or `numpy.ndarray` attributes, you can use the `...` filter which will match any node. For a more general filter you can pass a predicate function of the form:

```python
(path: Tuple[str, ...], value: Any) -> bool
```

To reconstruct the module from a set of substates, you can use `merge` as usual but passing the substates as variadic arguments:

```python
model = moduledef.merge(params, batch_stats, rest)
```

The same is true for `apply`.

```python
y, (state, moduledef) = moduledef.apply(params, batch_stats, rest)(x)
```

 Note that `apply` will return a single `state` object, if you need to re-partition the state you can use `State`'s own `partition` method:

```python
params, batch_stats, rest = state.partition("params", "batch_stats", ...)
```

Alternatively, if you are just interested in a subset of partitions, you can use the `State.filter` method which will not raise an error if some nodes are not matched by any filter:

```python
params, batch_stats = state.filter("params", "batch_stats")
```


### Stateful Transforms

Stateful transforms take a Module as their first argument, track changes in the state that occur within the transformation, and automatically propagate those changes to the input Module outside the transformation. In general, they behave as stateful functions with respect to the first argument.

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

# stateful update
train_step(model, x, y)
```

The most interesting aspect of this design is that the code appears very imperative, as the state is automatically propagated in and out of the transformations.

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
