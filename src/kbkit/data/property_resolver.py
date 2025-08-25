"""Alias maps for determining correct property names."""

import difflib
import json
from pathlib import Path

# Default alias map (can be extended or replaced)
ENERGY_ALIASES: dict[str, set[str]] = {
    "enthalpy": {"enthalpy", "enth", "h", "H"},
    "temperature": {"temperature", "temp", "t"},
    "volume": {"volume", "vol", "v"},
    "heat_capacity": {"cv", "c_v", "C_v", "Cv", "cp", "c_p", "C_p", "Cp", "heat_capacity", "heat_cap"},
    "pressure": {"pressure", "pres", "p"},
    "density": {"density", "rho"},
    "potential": {"potential_energy", "potential", "pe", "U"},
    "kinetic-en": {"kinetic_energy", "kinetic", "ke"},
    "total-energy": {"total_energy", "etot", "total", "E"},
    "time": {"time", "timestep", "dt"},
}


def load_gmx_unit_map() -> dict[str, str]:
    """
    Load GROMACS unit definitions from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file containing unit definitions.

    Returns
    -------
    dict
        Mapping of canonical property names to unit strings.
    """
    unit_path = Path(__file__).parent / "gmx_units.json"
    if not unit_path.exists():
        raise FileNotFoundError(f"Unit definition file not found: {unit_path}")
    with unit_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_attr_key(key: str, alias_map: dict[str, set[str]], cutoff: float = 0.6) -> str:
    """
    Resolve an attribute name to its canonical key using aliases and fuzzy matching.

    Parameters
    ----------
    value : str
        The attribute name to resolve.
    cutoff : float, optional
        Minimum similarity score to accept a match (default: 0.6).

    Returns
    -------
    str
        The canonical key corresponding to the input value.
    """
    key_lower = key.lower()
    match_to_key = {}
    best_match = None
    best_score = 0.0

    for canonical, aliases in alias_map.items():
        for alias in aliases:
            alias_lower = alias.lower()
            match_to_key[alias_lower] = canonical
            score = difflib.SequenceMatcher(None, key_lower, alias_lower).ratio()
            if score > best_score:
                best_score = score
                best_match = alias_lower

    if best_score >= cutoff and best_match:
        return match_to_key[best_match]
    raise KeyError(f"No close match found for '{key}' (best score: {best_score:.2f})")


def get_gmx_unit(name: str, alias_map: dict[str, set[str]] = ENERGY_ALIASES) -> str:
    """
    Retrieve the GROMACS unit for a given property name using alias resolution.

    Parameters
    ----------
    name : str
        Property name or alias.
    alias_map : dict
        Alias map to resolve the property name.
    unit_path : str
        Path to the JSON file containing unit definitions.

    Returns
    -------
    str
        Unit string (e.g., 'kJ/mol').
    """
    canonical = resolve_attr_key(name, alias_map)
    unit_map = load_gmx_unit_map()
    try:
        return unit_map[canonical]
    except KeyError as e:
        raise KeyError(f"No unit defined for canonical property '{canonical}' in {unit_map}") from e
