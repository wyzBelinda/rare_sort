"""Shared low-level helpers for parsing the wide-table cell values.

The wide table writes missing as the literal string "-" (never empty), per the
README. Everything here treats "-", "", None and NaN as missing and returns
None, so every calibrate() can stay simple and never blow up on missing data.
"""
from __future__ import annotations

import math
import re
from typing import Optional

_MISSING = {"", "-", "na", "n/a", "none", "."}


def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in _MISSING:
        return True
    return False


def to_float(x) -> Optional[float]:
    if is_missing(x):
        return None
    try:
        return float(str(x).strip())
    except (ValueError, TypeError):
        return None


def to_int(x) -> Optional[int]:
    f = to_float(x)
    return None if f is None else int(round(f))


def norm_token(x) -> str:
    """Lowercase, collapse any run of non-alphanumerics to a single '_'.

    Makes ClinVar text robust to the underscore->space conversion the wrapper
    does, and to punctuation differences across ClinVar releases.
    'criteria_provided,_multiple_submitters,_no_conflicts'  and
    'criteria provided, multiple submitters, no conflicts' both ->
    'criteria_provided_multiple_submitters_no_conflicts'.
    """
    if is_missing(x):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def split_terms(x) -> list[str]:
    """VEP joins multiple consequence terms with '&' (sometimes ',' or ';')."""
    if is_missing(x):
        return []
    return [norm_token(t) for t in re.split(r"[&,;]+", str(x)) if norm_token(t)]


def variant_key(chrom, pos, ref, alt) -> tuple:
    """Stable identity for a variant, independent of transcript row.

    Normalizes 'chr1' vs '1' and ref/alt case so truth labels and table rows
    join even if upstream conventions differ.
    """
    c = str(chrom).strip()
    c = c[3:] if c.lower().startswith("chr") else c
    return (c, str(pos).strip(), str(ref).strip().upper(), str(alt).strip().upper())
