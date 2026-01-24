"""String and data formatting."""

import difflib
import re
from re import Match

# Default alias map (can be extended or replaced)
ENERGY_ALIASES: dict[str, set[str]] = {
    "isothermal-compressibility": {"kappa", "kT", "kt", "isothermal_compressibility"},
    "cp": {"cp", "c_p", "C_p", "Cp", "heat_capacity", "heat_cap_cp"},
    "cv": {"cv", "c_v", "C_v", "Cv", "heat_capacity_v", "heat_cap_cv"},
    "time": {"time", "timestep", "dt"},
    "enthalpy": {"enthalpy", "enth", "h", "H"},
    "temperature": {"temperature", "temp", "t"},
    "volume": {"volume", "vol", "v"},
    "pressure": {"pressure", "pres", "p"},
    "density": {"density", "mass_volume"},
    "potential": {"potential_energy", "potential", "pe", "U"},
    "kinetic-en": {"kinetic_energy", "kinetic", "ke"},
    "total-energy": {"total_energy", "etot", "total", "E"},
    "number-density": {"number_density", "rho", "num_rho", "molec_per_volume"},
    "molar-volume": {"molar_volume", "mol_vol", "partial_volume"},
}


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

    else:
        formatted_key = key.replace(".", "").replace(" ", "-").replace("_", "-")
        parts_cap = [p.capitalize() for p in formatted_key.split("-")]
        return "-".join(parts_cap)


def resolve_units(requested: str, default: str) -> str:
    """
    Return the requested unit if provided, otherwise fall back to the default.

    Parameters
    ----------
    requested: str
        Desired units.
    default: str
        Units to fall back on.

    Returns
    -------
    str
        Units, either requested or default.
    """
    return requested if requested else default


def format_unit_str(text: str) -> str:
    """
    Convert a string representing mathematical expressions and units into LaTeX math format.

    Parameters
    ----------
    text : str
        The unit string to format.

    Returns
    -------
    str
        A LaTeX math string representing the units.
    """
    # check that object is string
    try:
        text = str(text)
    except TypeError as e:
        raise TypeError(f"Could not convert type {type(text)} to str.") from e

    def inverse_fix(match: Match[str]) -> str:
        """Replace /unit ** exponent with /unit^{exponent}."""
        unit = match.group(1)
        exp = match.group(2)
        return f"/{unit}^{{{exp}}}"

    # correct inverse unit format of first type
    text = re.sub(r"/\s*([a-zA-Z]+)\s*\*\*\s*(\d+)", inverse_fix, text)

    def inverse_unit_repl(match: Match[str]) -> str:
        """Inverse replacement for /unit^{exp} or /unitexp to unit^{-exp}."""
        unit = match.group(1)
        m_exp = re.match(r"^([a-zA-Z]+)\^\{(-?\d+)\}$", unit)
        if m_exp:
            letters, exponent = m_exp.groups()
            new_exp = str(-int(exponent))
            return rf"\text{{ }}\mathrm{{{letters}^{{{new_exp}}}}}"
        m_simple = re.match(r"^([a-zA-Z]+)(\d+)$", unit)
        if m_simple:
            letters, digits = m_simple.groups()
            return rf"\text{{ }}\mathrm{{{letters}^{{-{digits}}}}}"
        return rf"\text{{ }}\mathrm{{{unit}^{{-1}}}}"

    # replace /unit^{exp} to unit^{-exp}
    text = re.sub(r"/\s*([a-zA-Z0-9_\^\{\}]+)", inverse_unit_repl, text)

    # convert subscripts to _{val} FIRST
    text = re.sub(r"_(\(?[a-zA-Z0-9+\-*/=]+\)?)", r"_{\1}", text)

    # THEN convert superscripts **exp to ^{exp}
    text = re.sub(r"\*\*\s*([^\s_]+)", r"^{\1}", text)

    # wrap with $ if needed
    if not (text.startswith("$") and text.endswith("$")):
        text = f"${text}$"

    return text
