"""Configuration registry for swappable backends.

Encoder backends, fusion strategies, ensemble members, and rules engines are
registered by name and selected per deployment. New variants ship without
touching orchestration.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A namespaced name -> factory map.

    The factory signature is intentionally untyped (`Callable[..., T]`) — each
    registry instance documents its own constructor convention.
    """

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._items: dict[str, Callable[..., T]] = {}

    @property
    def namespace(self) -> str:
        return self._namespace

    def register(self, name: str, factory: Callable[..., T] | None = None):
        """Register a factory. Usable as a decorator or a direct call."""
        if factory is None:
            def decorator(fn: Callable[..., T]) -> Callable[..., T]:
                self.register(name, fn)
                return fn
            return decorator
        if name in self._items:
            raise ValueError(f"{self._namespace}: '{name}' already registered")
        self._items[name] = factory
        return factory

    def create(self, name: str, **kwargs) -> T:
        if name not in self._items:
            raise KeyError(
                f"{self._namespace}: '{name}' not registered "
                f"(available: {sorted(self._items)})"
            )
        return self._items[name](**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name in self._items
