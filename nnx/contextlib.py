import builtins
import dataclasses
import hashlib
import typing as tp

import jax
import jax.tree_util as jtu

from nnx import errors, tracers

KeyArray = jax.Array
Counts = tp.Tuple[int, ...]


def _stable_hash(data: Counts) -> int:
    hash_str = " ".join(str(x) for x in data)
    _hash = hashlib.blake2s(hash_str.encode())
    hash_bytes = _hash.digest()
    # uint32 is represented as 4 bytes in big endian
    return int.from_bytes(hash_bytes[:4], byteorder="big")


class ContextDef:
    __slots__ = ("_rng_counts", "_flags")

    def __init__(
        self,
        rng_counts: tp.Tuple[tp.Tuple[str, Counts], ...],
        flags: tp.Tuple[tp.Tuple[str, bool], ...],
    ):
        self._rng_counts = rng_counts
        self._flags = flags

    def merge(self, keys: tp.Mapping[str, KeyArray]) -> "Context":
        rngs = {
            name: RngStream(keys[name], count=0, count_path=count_path)
            for name, count_path in self._rng_counts
        }
        return Context(rngs=rngs, flags=dict(self._flags))


class PureContext(tp.Tuple[tp.Dict[str, KeyArray], ContextDef]):
    @classmethod
    def new(cls, keys: tp.Dict[str, KeyArray], contextdef: ContextDef):
        return cls((keys, contextdef))

    @property
    def keys(self) -> tp.Dict[str, KeyArray]:
        return self[0]

    @property
    def contextdef(self) -> ContextDef:
        return self[1]

    def merge(self):
        return self.contextdef.merge(self.keys)


def _pure_context_flatten(pure_context: PureContext):
    return tuple(pure_context), None


def _pure_context_unflatten(
    aux_data: None,
    children: tp.Tuple[tp.Dict[str, KeyArray], ContextDef],
) -> PureContext:
    return PureContext(children)


jtu.register_pytree_node(PureContext, _pure_context_flatten, _pure_context_unflatten)


@dataclasses.dataclass
class RngStream:
    key: KeyArray
    count: int = 0
    count_path: Counts = ()


class Context:
    __slots__ = ("_rngs", "_flags", "_trace_state")

    def __init__(
        self,
        rngs: tp.Mapping[str, RngStream],
        flags: tp.Mapping[str, bool],
    ):
        self._rngs = rngs
        self._flags = flags
        self._trace_state = tracers.TraceState()

    def has_rng(self, name: str) -> bool:
        return name in self._rngs

    def make_rng(self, name: str) -> KeyArray:
        if name not in self._rngs:
            raise ValueError(f"Unknown Rng Stream: {name}")
        elif not self._trace_state.is_valid():
            raise errors.TraceContextError(
                "Cannot use Context from a different trace level"
            )

        stream = self._rngs[name]
        fold_data = _stable_hash(stream.count_path + (stream.count,))
        stream.count += 1
        return jax.random.fold_in(stream.key, fold_data)

    def copy(self) -> "Context":
        return Context(rngs=self._rngs, flags=self._flags)

    def has_flag(self, name: str) -> bool:
        return name in self._flags

    def get_flag(self, name: str) -> tp.Optional[bool]:
        return self._flags.get(name, None)

    def partition(self) -> PureContext:
        if not self._trace_state.is_valid():
            raise errors.TraceContextError(
                "Cannot use Context from a different trace level"
            )

        def fork(stream) -> "RngStream":
            count_path = stream.count_path + (stream.count,)
            stream.count += 1
            return RngStream(stream.key, count_path=count_path)

        rngs = {name: fork(stream) for name, stream in self._rngs.items()}
        keys = {name: stream.key for name, stream in rngs.items()}
        rng_counts = tuple((name, stream.count_path) for name, stream in rngs.items())
        return PureContext.new(keys, ContextDef(rng_counts, tuple(self._flags.items())))


def context(
    params: tp.Union[int, KeyArray, RngStream, None] = None,
    *,
    flags: tp.Optional[tp.Mapping[str, bool]] = None,
    **rngs: tp.Union[int, KeyArray, RngStream],
) -> Context:
    _flags = flags or {}

    if params is not None:
        rngs["params"] = params

    _rngs = {
        name: RngStream(jax.random.PRNGKey(value))
        if isinstance(value, int)
        else RngStream(value)
        if isinstance(value, jax.Array)
        else value
        for name, value in rngs.items()
    }

    return Context(rngs=_rngs, flags=_flags)


if tp.TYPE_CHECKING:
    ellipsis = builtins.ellipsis
else:
    ellipsis = tp.Any

RngPredicate = tp.Callable[[str], bool]
RngFilterLiteral = tp.Union[str, RngPredicate, ellipsis, None]
RngFilter = tp.Union[RngFilterLiteral, tp.Sequence[RngFilterLiteral]]


def to_rng_predicate(filter: RngFilter) -> RngPredicate:
    if filter is None:
        return lambda _: False
    elif filter is ...:
        return lambda _: True
    elif callable(filter):
        return filter
    elif isinstance(filter, str):
        return lambda name: name == filter
    elif isinstance(filter, tp.Sequence):
        predicates = tuple(map(to_rng_predicate, filter))
        return lambda name: any(predicate(name) for predicate in predicates)
    else:
        raise TypeError(f"Invalid rng filter: {filter}")
