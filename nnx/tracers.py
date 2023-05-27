# Taken from flax/core/tracer.py 🏴‍☠️

import contextlib
import dataclasses
import threading
import typing as tp

import jax
import jax.core
from jax.core import MainTrace


@tp.runtime_checkable
class Tracer(tp.Protocol):
    _trace: jax.core.Trace


def current_jax_trace() -> MainTrace:
    """Returns the innermost Jax tracer."""
    return get_top_trace(())


def get_top_trace(pytree: tp.Union[tp.Any, Tracer]) -> MainTrace:
    """Returns the main top trace of a sequence of tracers."""
    if isinstance(pytree, Tracer):
        return pytree._trace.main

    return jax.core.find_top_trace(jax.tree_util.tree_leaves(pytree)).main


def get_all_traces(pytree: tp.Union[tp.Any, Tracer]) -> tp.Set[MainTrace]:
    """Returns True if all tracers have the same main trace."""
    if isinstance(pytree, Tracer):
        return {pytree._trace.main}
    else:
        return {
            trace._trace.main
            for trace in jax.tree_util.tree_leaves(pytree)
            if isinstance(trace, Tracer)
        }


def trace_level(main):
    """Returns the level of the trace of -infinity if it is None."""
    if main:
        return main.level
    return float("-inf")
