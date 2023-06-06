import typing as tp

from pure_pytree.partitioning import Partition, PartitionDef, Variable

A = tp.TypeVar("A", contravariant=True)
M = tp.TypeVar("M", bound="Module")


class Pytree:
    ...


class Module(Pytree):
    def replace(self: M, **kwargs: tp.Any) -> M:
        ...

    @tp.overload
    def partition(self: M) -> tp.Tuple[tp.Dict[str, Partition], PartitionDef[M]]:
        ...

    @tp.overload
    def partition(self: M, collection: str) -> tp.Tuple[Partition, PartitionDef[M]]:
        ...

    @tp.overload
    def partition(
        self: M, collection: str, *collections: str
    ) -> tp.Tuple[tp.Tuple[Partition, ...], PartitionDef[M]]:
        ...

    def partition(
        self: M, *collections: str
    ) -> tp.Tuple[
        tp.Union[tp.Dict[str, Partition], tp.Tuple[Partition, ...], Partition],
        PartitionDef[M],
    ]:
        ...

    @tp.overload
    def get_partition(self, collection: str) -> Partition:
        ...

    @tp.overload
    def get_partition(
        self, collection: str, *collections: str
    ) -> tp.Tuple[Partition, ...]:
        ...

    def get_partition(
        self, *collections: str
    ) -> tp.Union[Partition, tp.Tuple[Partition, ...]]:
        ...

    def merge(self: M, partition: Partition, *partitions: Partition) -> M:
        ...
