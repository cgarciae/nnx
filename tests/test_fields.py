import nnx


class TestFields:
    def test_dataclass(self):
        @nnx.dataclass
        class Foo:
            x: int
            y: int
            r: int = nnx.ref("state", init=False)

        foo = Foo(1, 2)

        assert foo.x == 1
        assert foo.y == 2
