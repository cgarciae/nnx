from nnx.module import Module
import typing as tp

A = tp.TypeVar("A")


class Map(Module, tp.Mapping[str, A]):
    def __init__(self, *args, **kwargs):
        vars(self).update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(vars(self))

    def __len__(self):
        return len(vars(self))


class Seq(Module, tp.Sequence[A]):
    def __init__(self, iterable: tp.Iterable[tp.Any]):
        for i, value in enumerate(iterable):
            setattr(self, str(i), value)

    def __getitem__(self, key: int):
        if key >= len(self):
            raise IndexError(f"Index {key} out of range for {self}")
        return getattr(self, str(key))

    def __iter__(self) -> tp.Iterator[A]:
        for i in range(len(self)):
            yield getattr(self, str(i))

    def __len__(self):
        return len(vars(self))
