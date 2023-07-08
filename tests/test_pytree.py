from typing import Generic, TypeVar

import jax
import pytest
from flax import serialization

import nnx


class TestPytree:
    def test_immutable_pytree(self):
        class Foo(nnx.Pytree):
            def __init__(self, y) -> None:
                self.x = 2
                self.y = nnx.Node(y)

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(
            AttributeError, match="is immutable, trying to update field"
        ):
            pytree.x = 4

    def test_immutable_pytree_dataclass(self):
        @nnx.dataclass(frozen=True)
        class Foo(nnx.Pytree):
            y: int = nnx.node_field()
            x: int = nnx.field(default=2)

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(AttributeError, match="cannot assign to field"):
            pytree.x = 4

    def test_jit(self):
        @nnx.dataclass
        class Foo(nnx.Pytree):
            a: int = nnx.node_field()
            b: int = nnx.field()

        module = Foo(a=1, b=2)

        @jax.jit
        def f(m: Foo):
            return m.a + m.b

        assert f(module) == 3

    def test_flax_serialization(self):
        class Bar(nnx.Pytree):
            def __init__(self, a, b):
                self.a = a
                self.b = nnx.Node(b)

        @nnx.dataclass
        class Foo(nnx.Pytree):
            bar: Bar
            c: int = nnx.node_field()
            d: int = nnx.field()

        foo: Foo = Foo(bar=Bar(a=1, b=2), c=3, d=4)

        state_dict = serialization.to_state_dict(foo)

        assert state_dict == {
            "bar": {
                "b": 2,
            },
            "c": 3,
        }

        state_dict["bar"]["b"] = 5

        foo = serialization.from_state_dict(foo, state_dict)

        assert foo.bar.b == 5

        del state_dict["bar"]["b"]

        with pytest.raises(ValueError, match="Missing field"):
            serialization.from_state_dict(foo, state_dict)

        state_dict["bar"]["b"] = 5

        # add unknown field
        state_dict["x"] = 6

        with pytest.raises(ValueError, match="Unknown field"):
            serialization.from_state_dict(foo, state_dict)

    def test_generics(self):
        T = TypeVar("T")

        class MyClass(nnx.Pytree, Generic[T]):
            def __init__(self, x: T):
                self.x = x

        MyClass[int]

    def test_key_paths(self):
        @nnx.dataclass
        class Bar(nnx.Pytree):
            a: int = nnx.node_field(default=1)
            b: int = nnx.field(default=2)

        @nnx.dataclass
        class Foo(nnx.Pytree):
            x: int = nnx.node_field(default=3)
            y: int = nnx.field(default=4)
            z: Bar = nnx.node_field(default_factory=Bar)

        foo = Foo()

        path_values, treedef = jax.tree_util.tree_flatten_with_path(foo)
        path_values = [(list(map(str, path)), value) for path, value in path_values]

        assert path_values[0] == ([".x", ".value"], 3)
        assert path_values[1] == ([".z", ".value", ".a", ".value"], 1)

    def test_replace_unknown_fields_error(self):
        class Foo(nnx.Pytree):
            pass

        with pytest.raises(ValueError, match="Trying to replace unknown fields"):
            Foo().replace(y=1)

    def test_dataclass_inheritance(self):
        @nnx.dataclass
        class A(nnx.Pytree):
            a: int = nnx.node_field(default=1)
            b: int = nnx.field(default=2)

        @nnx.dataclass
        class B(A):
            c: int = nnx.node_field(default=3)

        pytree = B()
        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [1, 3]

    def test_pytree_with_new(self):
        class A(nnx.Pytree):
            def __init__(self, a):
                self.a = a

            def __new__(cls, a):
                return super().__new__(cls)

        pytree = A(a=1)

        pytree = jax.tree_map(lambda x: x * 2, pytree)

    def test_deterministic_order(self):
        class A(nnx.Pytree):
            def __init__(self, order: bool):
                if order:
                    self.a = 1
                    self.b = 2
                else:
                    self.b = 2
                    self.a = 1

        p1 = A(order=True)
        p2 = A(order=False)

        leaves1 = jax.tree_util.tree_leaves(p1)
        leaves2 = jax.tree_util.tree_leaves(p2)

        assert leaves1 == leaves2


class TestMutablePytree:
    def test_pytree(self):
        class Foo(nnx.Pytree, mutable=True):
            def __init__(self, y) -> None:
                self.x = 2
                self.y = nnx.Node(y)

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        # test mutation
        pytree.x = 4
        assert pytree.x == 4

    def test_no_new_fields_after_init(self):
        class Foo(nnx.Pytree, mutable=True):
            def __init__(self, x):
                self.x = nnx.Node(x)

        foo = Foo(x=1)
        foo.x = 2

        with pytest.raises(AttributeError, match=r"Cannot add new fields to"):
            foo.y = 2

    def test_pytree_dataclass(self):
        @nnx.dataclass
        class Foo(nnx.Pytree, mutable=True):
            y: int = nnx.node_field()
            x: int = nnx.field(default=2)

        pytree: Foo = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        # test mutation
        pytree.x = 4
        assert pytree.x == 4
