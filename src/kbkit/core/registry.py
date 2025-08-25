"""Semantic wrapper around discovered systems."""

from collections import defaultdict
from kbkit.schema.system_metadata import SystemMetadata


class SystemRegistry:
    def __init__(self, systems: list[SystemMetadata]) -> None:
        self._systems = systems
        self._by_name = {s.name: s for s in systems}
        self._by_kind = self._group_by_kind(systems)

    def _group_by_kind(self, systems: SystemMetadata) -> dict:
        """group the systems by kind."""
        grouped = defaultdict(list)
        for s in systems:
            grouped[s.kind].append(s)
        return grouped 
    
    def get(self, name: str) -> SystemMetadata:
        """Get system by name."""
        return self._by_name[name]
    
    def filter_by_kind(self, kind: str) -> list[SystemMetadata]:
        """Get list of systems by kind."""
        return self._by_kind.get(kind, [])
    
    def all(self) -> list[SystemMetadata]:
        """Get all systems."""
        return self._systems
    
    def get_idx(self, name: str) -> int:
        """Get system index in registry list."""
        systems_list = list(self._by_name.keys())
        return systems_list.index(name)
    
    def __iter__(self) -> None:
        """Allows you to loop over systems directly.
        
        Examples
        --------
        >>> from kbkit.core import SystemRegistry
        >>> registry = SystemRegistry(systems=[sys_1, sys_2])
        >>> for system in registry:
        ...     print(system.name)
        """
        return iter(self._systems)
    
    def __len__(self) -> int:
        """Get the number of systems in registry.
        
        Examples
        --------
        >>> registry = SystemRegistry([SystemMetadata(...), SystemMetadata(...)])
        >>> len(registry)
        2
        """
        return len(self._systems)
    
