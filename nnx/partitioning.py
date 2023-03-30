import dataclasses
import refx


@dataclasses.dataclass
class Is:
    collection: str

    def __call__(self, x):
        return isinstance(x, refx.Referential) and x.collection == self.collection
