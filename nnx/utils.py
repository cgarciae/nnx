import dataclasses
import inspect
import typing as tp

A = tp.TypeVar("A")


def first_from(*args: tp.Optional[A]) -> A:
    """Return the first non-None argument."""
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError("No non-None arguments found.")


def has_keyword_arg(func: tp.Callable[..., tp.Any], name: str) -> bool:
    """Return True if func has keyword-only arguments with the given name."""
    return any(
        param.name == name
        and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
        for param in inspect.signature(func).parameters.values()
    )


class _ProxyContext(tp.Protocol):
    def __call__(self, __fn: tp.Callable[..., tp.Any], *args, **kwargs) -> tp.Any:
        ...


@dataclasses.dataclass
class CallableProxy:
    _proxy_context: _ProxyContext
    _proxy_callable: tp.Callable[..., tp.Any]

    def __call__(self, *args, **kwargs):
        return self._proxy_context(self._proxy_callable, *args, **kwargs)

    def __getattr__(self, name) -> "CallableProxy":
        return CallableProxy(self._proxy_context, getattr(self._proxy_callable, name))

    def __getitem__(self, key) -> "CallableProxy":
        return CallableProxy(self._proxy_context, self._proxy_callable[key])


def _identity(x):
    return x


@dataclasses.dataclass
class DelayedAccessor:
    accessor: tp.Callable[[tp.Any], tp.Any] = _identity

    def __call__(self, x):
        return self.accessor(x)

    def __getattr__(self, name):
        return DelayedAccessor(lambda x: getattr(x, name))

    def __getitem__(self, key):
        return DelayedAccessor(lambda x: x[key])
