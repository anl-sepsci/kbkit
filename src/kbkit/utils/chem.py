"""Contains general-purpose chemical utilities such as element lookup."""

from rdkit.Chem import GetPeriodicTable

MAX_SYMBOL_LENGTH = 2


def is_valid_element(symbol: str) -> bool:
    """
    Check if a string is a valid chemical element symbol.

    Parameters
    ----------
    symbol : str
        Element symbol to validate (e.g., 'C', 'Na', 'Cl').

    Returns
    -------
    bool
        True if valid element, False otherwise.
    """
    if not symbol or not isinstance(symbol, str):
        return False

    symbol = symbol.strip().capitalize()
    if len(symbol) > MAX_SYMBOL_LENGTH:
        symbol = symbol[:2]

    ptable = GetPeriodicTable()
    return ptable.GetAtomicNumber(symbol) > 0


def get_atomic_number(symbol: str) -> int:
    """
    Return atomic number of a valid element symbol.

    Parameters
    ----------
    symbol : str
        Valid element symbol (e.g., 'C', 'Na', 'Cl').

    Returns
    -------
    int
        Atomic number of the element.
    """
    symbol = symbol.strip().capitalize()
    # checks that symbol is a valid element
    if is_valid_element(symbol):
        ptable = GetPeriodicTable()
        return ptable.GetAtomicNumber(symbol)
    else:
        raise ValueError(f"Symbol '{symbol}' is not a valid element.")
