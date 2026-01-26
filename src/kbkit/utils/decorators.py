"""Decorators for returning :class:`~kbkit.schema.property_result.PropertyResult` object or its `value` attribute."""

import inspect
from functools import wraps
from typing import Any, Callable

import numpy as np

from kbkit.schema.property_result import PropertyResult


def cached_property_result(default_units: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., PropertyResult]]:
    """Decorator factory for caching PropertyResult calculations.

    Parameters
    ----------
    default_units: str
        Default units for the property

    Returns
    -------
    Callable
        A decorator that wraps methods to return PropertyResult objects
    """

    def decorator(func) -> Callable[..., PropertyResult]:
        # Get function signature to access default values
        """
        Decorator for caching PropertyResult calculations.

        Parameters
        ----------
        func: Callable[..., Any]
            Method to be wrapped.

        Returns
        -------
        Callable[..., PropertyResult]
            A decorator that wraps methods to return PropertyResult objects.

        Notes
        -----
        * The decorator caches the result of the function call.
        * If the function call has a 'name' parameter, it is used as the property name.
        * If the function call does not have a 'name' parameter, the function name is used as the property name.
        * The property type is inferred from the function name by splitting the name on '_' and taking the first part.
        * The decorator preserves function metadata.
        * The decorator applies unit conversion if the 'units' parameter is provided.
        """
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self, units: str | None = None, **kwargs) -> PropertyResult:
            # Merge defaults with provided kwargs
            """
            Decorator for caching PropertyResult calculations.

            Parameters
            ----------
            self : Any
                Self parameter of the method to be wrapped.
            units : str | None
                Units for the property.
            **kwargs : dict
                Additional keyword arguments passed to the method.

            Returns
            -------
            PropertyResult
                The cached PropertyResult object, or a new one if not cached.
            """
            bound_args = sig.bind_partial(self, **kwargs)
            bound_args.apply_defaults()

            # Extract all arguments (excluding 'self' and 'units')
            all_kwargs = {k: v for k, v in bound_args.arguments.items() if k not in ("self", "units")}

            if "name" in all_kwargs:
                property_name = all_kwargs["name"]
                property_type = str(func.__name__).split("_")[0]
            else:
                property_name = func.__name__
                property_type = None

            func_meta = {k: v for k, v in all_kwargs.items() if k not in ("name", "avg")}

            cache_key = (
                property_name,
                property_type,
                *(f"{k}={v}" for k, v in sorted(all_kwargs.items()))
            )

            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                return cached_result.to(units) if units else cached_result

            # Determine units to use for calculation
            calc_units = units or default_units

            # Pass units to the function if it needs them
            if calc_units:
                all_kwargs["units"] = calc_units

            # Call the original function - pass all_kwargs to include defaults
            values = func(self, **all_kwargs)

            # Automatically wrap in PropertyResult
            result = PropertyResult(
                name=property_name, value=values, property_type=property_type, units=calc_units, metadata=func_meta
            )

            self._cache[cache_key] = result
            return result.to(units) if units else result

        return wrapper

    return decorator


def cached_property_value(default_units: str | None = None)  -> Callable[[Callable[..., Any]], Callable[..., np.ndarray]]:
    """
    Decorator factory for caching PropertyResult calculations and returning values.

    Parameters
    ----------
    default_units: str
        Default units for the property

    Returns
    -------
    Callable
        A decorator that wraps methods to return the value attribute of PropertyResult
    """
    def decorator(func) -> Callable[..., Any]:
        # Get function signature to access default values
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self, units: str | None = None, **kwargs) -> np.ndarray:
            # Merge defaults with provided kwargs
            """
            Wrapper function for caching PropertyResult calculations and returning values.

            Parameters
            ----------
            self : object
            units : str | None, optional
                Units to convert the result to, by default
            **kwargs : dict
                Additional keyword arguments to pass to the decorated function

            Returns
            -------
            Any
                The value attribute of the cached PropertyResult object
            """
            bound_args = sig.bind_partial(self, **kwargs)
            bound_args.apply_defaults()

            # Extract all arguments (excluding 'self' and 'units')
            all_kwargs = {k: v for k, v in bound_args.arguments.items() if k not in ("self", "units")}

            func_meta = {k: v for k, v in all_kwargs.items() if k not in ("name", "avg")}

            if "name" in all_kwargs:
                property_name = all_kwargs["name"]
                property_type = str(func.__name__).split("_")[0]
                cache_key = (property_name, property_type, *(f"{k}={v}" for k, v in sorted(all_kwargs.items())))
            else:
                property_name = func.__name__
                property_type = None
                cache_key = property_name

            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                result = cached_result.to(units) if units else cached_result

            else:
                # Determine units to use for calculation
                calc_units = units or default_units

                # Pass units to the function if it needs them
                if calc_units:
                    all_kwargs["units"] = calc_units

                # Call the original function - pass all_kwargs to include defaults
                values = func(self, **all_kwargs)

                # Automatically wrap in PropertyResult
                result = PropertyResult(
                    name=property_name, value=values, property_type=property_type, units=calc_units, metadata=func_meta
                )

                self._cache[cache_key] = result
                result = result.to(units) if units else result

            return result.value

        return wrapper

    return decorator
