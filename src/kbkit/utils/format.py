"""String and data formatting."""

import difflib
import re

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
    "kinetic en.": {"kinetic_energy", "kinetic", "ke"},
    "total energy": {"total_energy", "etot", "total", "E"},
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

    # 1. Handle explicit subscript syntax: X_(stuff) -> X_{stuff}
    text = re.sub(r"_\(([^)]+)\)", r"_{\1}", text)

    # 2. Handle implicit subscripts: X_num -> X_{num} (e.g., H_2 -> H_{2})
    text = re.sub(r"_(\d+)", r"_{\1}", text)

    # 3. Normalize: convert ** to ^
    text = text.replace("**", "^")

    # 4. Ensure all exponents use braces: X^num -> X^{num}
    text = re.sub(r"\^(\d+)", r"^{\1}", text)

    # 5. Handle division: /base^exp -> base^{-exp}
    def inverse_repl(match):
        base = match.group(1)
        exponent = match.group(2)
        if exponent:
            clean_exp = exponent.replace("{", "").replace("}", "")
            return rf"\,{base}^{{-{clean_exp}}}"
        return rf"\,{base}^{{-1}}"

    text = re.sub(r"/\s*([a-zA-Z]+)(?:\^\{?(-?\d+)\}?)?", inverse_repl, text)

    # 6. Final LaTeX cleanup
    text = text.replace("*", r"\,")

    # 7. Wrap in $ if not already
    if not (text.startswith("$") and text.endswith("$")):
        text = f"${text}$"

    return text
