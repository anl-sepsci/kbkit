"""String and data formatting."""

import re
from re import Match


def resolve_units(requested: str, default: str) -> str:
    """Return the requested unit if provided, otherwise fall back to the default."""
    return requested if requested else default


def _str_to_latex_math(text: str) -> str:
    """Convert a string representing mathematical expressions and units into LaTeX math format."""
    try:

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

        # convert superscripts **exp to ^{exp}
        text = re.sub(r"\*\*\s*(\(?[^\s\)]+(?:[^\s]*?)\)?)", r"^{\1}", text)

        # convert subscripts to _{val}
        text = re.sub(r"_(\(?[a-zA-Z0-9+\-*/=]+\)?)", r"_{\1}", text)

        # wrap with $ if needed
        if not (text.startswith("$") and text.endswith("$")):
            text = f"${text}$"

        return text

    except Exception as e:
        raise RuntimeError(f"Error converting string to LaTeX math: {e}") from e


def format_unit_str(text: str) -> str:
    """Format a unit string: convert unit words to symbols.

    If given a Pint Quantity, extract its units and format them in short form.
    The result is converted to LaTeX-friendly math for plotting.

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

    # format text for plotting
    unit_str = _str_to_latex_math(text)
    return unit_str
