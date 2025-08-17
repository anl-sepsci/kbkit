"""Alias maps for determining correct property names."""

import difflib

gmx_unit_map = {
    "enthalpy": "kJ/mol",
    "temperature": "kelvin",
    "volume": "nm^3",
    "heat_capacity": "kJ/mol/K",  # For fluct_prop Cp
    "pressure": "bar",
    "density": "kg/m^3",
    "potential": "kJ/mol",
    "kinetic-en": "kJ/mol",
    "total-energy": "kJ/mol",
    "time": "ps",
}

energy_aliases = {
    "enthalpy": {"enthalpy", "enth", "h", "H"},
    "temperature": {"temperature", "temp", "t"},
    "volume": {"volume", "vol", "v"},
    "heat_capacity": {
        "cv",
        "c_v",
        "C_v",
        "Cv",
        "cp",
        "c_p",
        "C_p",
        "Cp",
        "heat_capacity",
        "heat_cap",
    },
    "pressure": {"pressure", "pres", "p"},
    "density": {"density", "rho"},
    "potential": {"potential_energy", "potential", "pe", "U"},
    "kinetic-en": {"kinetic_energy", "kinetic", "ke"},
    "total-energy": {"total_energy", "etot", "total", "E"},
}

kb_aliases = {
    "lngamma": {"lngamma", "lngammas", "ln_gamma", "ln_gammas", "lng", "gammas"},
    "dlngamma": {
        "dlngamma",
        "dlngammas",
        "dln_gamma",
        "dln_gammas",
        "dln_gamma_dxs",
        "dln_gammas_dxs",
        "dlng_dx",
    },
    "lngamma_fits": {
        "lngamma_fits",
        "lngammas_fits",
        "lngamma_fns",
        "lng_fits",
        "gamma_fits",
        "fitted_gammas",
    },
    "dlngamma_fits": {
        "dlngamma_fits",
        "dlngamma_fns",
        "dlng_fits",
        "dlng_dx_fits",
        "dlngamma_dxs_fits",
    },
    "mixing": {"mixing", "mix", "mix_compare", "thermo_mixing"},
    "excess": {"excess", "ex", "ex_compare", "thermo_excess"},
    "ge": {"gibbs_excess", "ge", "excess_energy"},
    "gm": {"gibbs_mixing", "gm", "mixing_energy"},
    "hmix": {"mixing_enthalpy", "enthalpy", "h", "hmix", "he"},
    "se": {"excess_entropy", "se", "entropy", "s", "s_ex", "sex"},
    "kbi": {"kbi", "kbis", "kbintegrals", "kirkwood-buff"},
    "rdf": {"rdf", "gr", "g(r)"},
    "i0": {"i0", "saxs_i0", "saxs_intensity", "saxs_i0_conc", "saxs_i0_density"},
    "det_h": {"det_h", "hessian", "det_hessian", "h_ij", "det_h_ij", "d2gm"},
}


def get_gmx_unit(name: str) -> str:
    """
    Retrieve the default GROMACS units for a given property.

    Parameters
    ----------
    name: str
        Property to return the GROMACS units for.

    Returns
    -------
    str
        Default GROMACS units for name.
    """
    prop = resolve_attr_key(name, energy_aliases)
    try:
        return gmx_unit_map[prop]
    except KeyError as e:
        raise KeyError(f"Key '{prop}' does not exist in _gmx_unit_map: {gmx_unit_map.keys()}") from e


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
    # validate input
    try:
        value = key.lower()
    except AttributeError as e:
        raise TypeError(f"Input value must be a string, got {type(value)}: {value!r}") from e

    best_match = None
    best_score = 0.0
    match_to_key = {}

    # Flatten all aliases to map them back to their canonical key
    for canonical_key, aliases in alias_map.items():
        # validate aliases type
        if not isinstance(aliases, (list, tuple, set)):
            raise TypeError(f"Aliases for key '{canonical_key}' must be list/tuple/set, got {type(aliases)}")

        # get the score for the alias and update the best score and match
        for alias in aliases:
            # check that alias is of the correct type
            try:
                alias_lower = alias.lower()
            except AttributeError as e:
                raise TypeError(f"Alias must be a string, got {type(alias)}: {alias!r}") from e

            match_to_key[alias_lower] = canonical_key.lower()
            score = difflib.SequenceMatcher(None, value, alias_lower).ratio()
            if score > best_score:
                best_score = score
                best_match = alias_lower

    # check that score obeys cuttoff mark
    if best_score >= cutoff:
        if isinstance(best_match, str):
            return match_to_key[best_match]
        else:
            raise TypeError(f"Unexpected key type detected ({type(best_match)}), expected str.")
    else:
        raise KeyError(f"No close match found for '{value}' (best score: {best_score:.2f})")
