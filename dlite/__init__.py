"""DLite inference API, lazily loaded for lightweight data/benchmark commands."""
from importlib import import_module

__all__ = ["DLiteDraftModel", "DLiteEngine", "GenerationResult"]


def __getattr__(name):
    module = {"DLiteDraftModel": ".modeling", "DLiteEngine": ".engine", "GenerationResult": ".engine"}.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
