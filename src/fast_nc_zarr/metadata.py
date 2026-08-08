from __future__ import annotations

from collections.abc import Mapping

import xarray as xr

_SINGLE_REFERENCE_ATTRIBUTES = (
    "bounds",
    "climatology",
    "geometry",
    "grid_mapping",
)
_LIST_REFERENCE_ATTRIBUTES = (
    "ancillary_variables",
    "coordinates",
    "node_coordinates",
)
_KEYED_REFERENCE_ATTRIBUTES = (
    "cell_measures",
    "formula_terms",
)


def _resolved_name(name: str, renames: Mapping[str, str]) -> str:
    return str(renames.get(name, name))


def _sanitize_single(value: object, available: set[str], renames: Mapping[str, str]) -> str | None:
    tokens = str(value).split()
    if len(tokens) != 1:
        return None
    name = _resolved_name(tokens[0], renames)
    return name if name in available else None


def _sanitize_list(value: object, available: set[str], renames: Mapping[str, str]) -> str | None:
    names = []
    for token in str(value).split():
        name = _resolved_name(token, renames)
        if name in available and name not in names:
            names.append(name)
    return " ".join(names) or None


def _sanitize_keyed(value: object, available: set[str], renames: Mapping[str, str]) -> str | None:
    tokens = str(value).split()
    if len(tokens) % 2:
        return None
    entries = []
    for index in range(0, len(tokens), 2):
        label = tokens[index]
        if not label.endswith(":"):
            return None
        name = _resolved_name(tokens[index + 1], renames)
        if name in available:
            entries.extend((label, name))
    return " ".join(entries) or None


def sanitize_cf_references(
    dataset: xr.Dataset,
    *,
    renames: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Remove or rewrite CF attributes that reference absent variables.

    Attribute dictionaries are replaced in place; array data is never copied.
    """

    result = dataset
    rename_map = dict(renames or {})
    available = set(result.variables)
    for variable in result.variables.values():
        attrs = dict(variable.attrs)
        for key in _SINGLE_REFERENCE_ATTRIBUTES:
            if key not in attrs:
                continue
            value = _sanitize_single(attrs[key], available, rename_map)
            if value is None:
                attrs.pop(key, None)
            else:
                attrs[key] = value
        for key in _LIST_REFERENCE_ATTRIBUTES:
            if key not in attrs:
                continue
            value = _sanitize_list(attrs[key], available, rename_map)
            if value is None:
                attrs.pop(key, None)
            else:
                attrs[key] = value
        for key in _KEYED_REFERENCE_ATTRIBUTES:
            if key not in attrs:
                continue
            value = _sanitize_keyed(attrs[key], available, rename_map)
            if value is None:
                attrs.pop(key, None)
            else:
                attrs[key] = value
        variable.attrs = attrs
    return result
