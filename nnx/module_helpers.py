from nnx.module import Module
import typing as tp

A = tp.TypeVar("A")


class Map(Module, tp.Mapping[str, A]):
    def __init__(self, *args, **kwargs):
        vars(self).update(*args, **kwargs)

    def __getitem__(self, key) -> A:
        return vars(self)[key]

    def __iter__(self) -> tp.Iterator[str]:
        return iter(vars(self))

    def __len__(self) -> int:
        return len(vars(self))


class Seq(Module, tp.Sequence[A]):
    def __init__(self, iterable: tp.Iterable[A]):
        for i, value in enumerate(iterable):
            vars(self)[str(i)] = value

    def __getitem__(self, key: int) -> A:
        if key >= len(self):
            raise IndexError(f"Index {key} out of range for {self}")
        return vars(self)[str(key)]

    def __iter__(self) -> tp.Iterator[A]:
        for i in range(len(self)):
            yield vars(self)[str(i)]

    def __len__(self) -> int:
        return len(vars(self))
