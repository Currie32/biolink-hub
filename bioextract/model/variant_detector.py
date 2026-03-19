from __future__ import annotations

"""Regex-based variant detection for biomedical text.

Detects variant mentions via pattern matching — no LLM or dictionary needed.
Returns entity dicts with exact character spans from the text.
"""

import re


# --- Variant patterns (ordered from most specific to least) ---

_PATTERNS = [
    # DNA substitutions: c.1799T>A, c.444-62C>A, g.140453136A>T
    re.compile(r'[cg]\.\d+[-+]?\d*[ACGT]>[ACGT]'),

    # RS numbers: rs113488022, rs1799971
    re.compile(r'rs\d{3,}'),

    # Protein substitutions (3-letter): p.Val600Glu, p.Arg132His
    re.compile(r'p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}'),

    # Protein substitutions (1-letter with p. prefix): p.V600E
    re.compile(r'p\.[A-Z]\d+[A-Z]'),

    # Allele/star notations: NQO1*2, CYP2D6*4
    re.compile(r'[A-Z][A-Z0-9]{1,10}\*\d+'),

    # Codon substitutions: G-->A, C-->T (with arrow notation)
    re.compile(r'[ACGT]\s*-+>\s*[ACGT]\s+substitution'),

    # Protein substitutions (1-letter, no prefix): V600E, R132H, A118G, N40D
    # Must be uppercase letter + digits + uppercase letter, at word boundary
    # Require at least 2 digits to avoid matching gene names like "V1"
    re.compile(r'\b[ACDEFGHIKLMNPQRSTVWY]\d{2,}[ACDEFGHIKLMNPQRSTVWY]\b'),

    # Amino acid description patterns:
    # "valine [V] to methionine [M] substitution at position 244"
    re.compile(
        r'[a-z]+\s+\[[A-Z]\]\s+to\s+[a-z]+\s+\[[A-Z]\]\s+'
        r'(?:substitution|change|mutation)\s+at\s+position\s+\d+',
        re.IGNORECASE,
    ),
]


def detect_variants(text: str) -> list[dict]:
    """Detect variant mentions in text via regex patterns.

    Returns list of {"text": str, "type": "VARIANT", "start": int, "end": int}.
    Each match uses the exact span from the input text.
    Deduplicates overlapping matches (prefers longer).
    """
    matches = []
    for pattern in _PATTERNS:
        for m in pattern.finditer(text):
            matches.append({
                "text": m.group(),
                "type": "VARIANT",
                "start": m.start(),
                "end": m.end(),
            })

    if not matches:
        return matches

    # Deduplicate: remove matches whose spans are fully contained in a longer match
    matches.sort(key=lambda m: (m["start"], -m["end"]))
    deduped = []
    for m in matches:
        # Check if this match is subsumed by any already-kept match
        subsumed = False
        for kept in deduped:
            if m["start"] >= kept["start"] and m["end"] <= kept["end"]:
                subsumed = True
                break
        if not subsumed:
            deduped.append(m)

    return deduped
