from abc import abstractmethod
import dataclasses
import threading
import typing as tp
import contextlib


@dataclasses.dataclass
class Context(threading.local):
    indent_stack: tp.List[str] = dataclasses.field(default_factory=lambda: [""])


CONTEXT = Context()


@dataclasses.dataclass
class Config:
    type: tp.Union[str, type]
    parens_left: str = "("
    parens_right: str = ")"
    value_sep: str = "="
    elem_indent: str = "  "
    empty_repr: str = ""


@dataclasses.dataclass
class Elem:
    key: str
    value: tp.Union[str, tp.Any]
    start: str = ""
    end: str = ""


class Representable:
    __slots__ = ()

    @abstractmethod
    def __nnx_repr__(self) -> tp.Iterator[tp.Union[Config, Elem]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return get_repr(self)


@contextlib.contextmanager
def add_indent(indent: str) -> tp.Iterator[None]:
    CONTEXT.indent_stack.append(CONTEXT.indent_stack[-1] + indent)

    try:
        yield
    finally:
        CONTEXT.indent_stack.pop()


def get_indent() -> str:
    return CONTEXT.indent_stack[-1]


def get_repr(obj: Representable) -> str:
    if not isinstance(obj, Representable):
        raise TypeError(f"Object {obj!r} is not representable")

    iterator = obj.__nnx_repr__()
    config = next(iterator)
    if not isinstance(config, Config):
        raise TypeError(f"First item must be Config, got {type(config).__name__}")

    def _repr_elem(elem: tp.Any) -> str:
        if not isinstance(elem, Elem):
            raise TypeError(f"Item must be Elem, got {type(elem).__name__}")

        value = elem.value if isinstance(elem.value, str) else repr(elem.value)

        if "\n" in value and not isinstance(elem.value, Representable):
            value = value.replace("\n", "\n" + get_indent())

        return (
            f"{get_indent()}{elem.start}{elem.key}{config.value_sep}{value}{elem.end}"
        )

    with add_indent(config.elem_indent):
        elems = list(map(_repr_elem, iterator))
    elems = ",\n".join(elems)

    if elems:
        elems = "\n" + elems + "\n" + get_indent()
    else:
        elems = config.empty_repr

    type_repr = config.type if isinstance(config.type, str) else config.type.__name__

    return f"{type_repr}{config.parens_left}{elems}{config.parens_right}"
