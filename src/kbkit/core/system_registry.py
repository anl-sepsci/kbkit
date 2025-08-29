"""Semantic wrapper around discovered systems."""

from collections import defaultdict
from typing import Iterator

from kbkit.schema.system_metadata import SystemMetadata


class SystemRegistry:
    """
    Registry of discovered molecular systems with semantic access patterns.

    Stores and organizes SystemMetadata objects by name and kind, enabling
    reproducible filtering, indexing, and iteration across pure and mixture systems.

    Parameters
    ----------
    systems : list[SystemMetadata]
        List of discovered systems to register.
    """

    def __init__(self, systems: list[SystemMetadata]) -> None:
        self._systems = systems
        self._by_name = {s.name: s for s in systems}
        self._by_kind = self._group_by_kind(systems)

    def _group_by_kind(self, systems: list[SystemMetadata]) -> dict[str, list[SystemMetadata]]:
        """
        Group systems by their kind attribute.

        Parameters
        ----------
        systems : list[SystemMetadata]
            List of systems to group.

        Returns
        -------
        dict[str, list[SystemMetadata]]
            Dictionary mapping kind to list of systems.
        """
        grouped = defaultdict(list)
        for s in systems:
            grouped[s.kind].append(s)
        return grouped

    def get(self, name: str) -> SystemMetadata:
        """
        Retrieve a system by its name.

        Parameters
        ----------
        name : str
            Name of the system to retrieve.

        Returns
        -------
        SystemMetadata
            Corresponding system metadata object.
        """
        return self._by_name[name]

    def filter_by_kind(self, kind: str) -> list[SystemMetadata]:
        """
        Retrieve all systems of a given kind.

        Parameters
        ----------
        kind : str
            Kind of system to filter by (e.g., "pure", "mixture").

        Returns
        -------
        list[SystemMetadata]
            List of systems matching the specified kind.
        """
        return self._by_kind.get(kind, [])

    def all(self) -> list[SystemMetadata]:
        """
        Return all registered systems.

        Returns
        -------
        list[SystemMetadata]
            Full list of systems in the registry.
        """
        return self._systems

    def get_idx(self, name: str) -> int:
        """
        Get the index of a system by name in the registry list.

        Parameters
        ----------
        name : str
            Name of the system.

        Returns
        -------
        int
            Index of the system in the registry.
        """
        systems_list = list(self._by_name.keys())
        return systems_list.index(name)

    def __iter__(self) -> Iterator[SystemMetadata]:
        """
        Enable iteration over registered systems.

        Returns
        -------
        Iterator[SystemMetadata]
            Iterator over all systems.

        Examples
        --------
        >>> for system in registry:
        ...     print(system.name)
        """
        return iter(self._systems)

    def __len__(self) -> int:
        """
        Return the number of systems in the registry.

        Returns
        -------
        int
            Total number of registered systems.

        Examples
        --------
        >>> len(registry)
        5
        """
        return len(self._systems)
