"""Decorators for returning PropertyResult object or its `value` attribute."""

import inspect
from functools import wraps

from kbkit.schema.property_result import PropertyResult


def cached_property_result(default_units: str | None = None):
    """Decorator factory for caching PropertyResult calculations."""

    def decorator(func):
        # Get function signature to access default values
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self, units: str | None = None, **kwargs):
            # Merge defaults with provided kwargs
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


def cached_property_value(default_units: str | None = None):
    """Decorator factory for caching PropertyResult calculations."""

    def decorator(func):
        # Get function signature to access default values
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self, units: str | None = None, **kwargs):
            # Merge defaults with provided kwargs
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
