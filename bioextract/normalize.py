"""Entity normalization pipeline.

After the model extracts raw entity mentions, this pipeline resolves them
to canonical IDs using:
1. Exact dictionary match (fast path)
2. Abbreviation disambiguation for ambiguous terms
3. FTS search with type filter
4. FTS search WITHOUT type filter (cross-type fallback — DISEASE in HPO as PHENOTYPE, etc.)
5. Text normalization (plurals, parentheses, prefixes/suffixes)
6. Candidate ranking by string similarity + type consistency
"""

import logging
import re
from dataclasses import dataclass

from .dictionaries.lookup import DictionaryLookup, DictMatch, _normalize_type
from .schema import ExtractedEntity, ExtractionResult

logger = logging.getLogger(__name__)

# Pattern to find inline abbreviation definitions: "full name (ABBR)"
# Matches both "long QT syndrome (LQTS)" and "Deoxyguanosine kinase (dGK)"
_ABBREV_DEFINITION_RE = re.compile(
    r'([A-Z][a-zA-Z\s\-\(\)]{3,50}?)\s*\(([A-Za-z][A-Za-z0-9]{1,10})\)'
)

# Types that should cross-match in dictionary lookups
# e.g., DISEASE entities should also match PHENOTYPE dictionary entries
COMPATIBLE_TYPES = {
    "DISEASE": ["DISEASE", "PHENOTYPE"],
    "PHENOTYPE": ["PHENOTYPE", "DISEASE"],
    "GENE": ["GENE", "PROTEIN"],
    "PROTEIN": ["GENE", "PROTEIN"],
    "CHEMICAL": ["CHEMICAL", "DRUG"],
    "DRUG": ["CHEMICAL", "DRUG"],
}

# Common unambiguous biomedical abbreviations -> canonical expansion
# These don't need context disambiguation — they have one dominant meaning
COMMON_ABBREVIATIONS = {
    "HBV": "hepatitis B virus",
    "HCV": "hepatitis C virus",
    "HIV": "human immunodeficiency virus",
    "HPV": "human papillomavirus",
    "EBV": "Epstein-Barr virus",
    "CMV": "cytomegalovirus",
    "RSV": "respiratory syncytial virus",
    "TB": "tuberculosis",
    "COPD": "chronic obstructive pulmonary disease",
    "CHF": "congestive heart failure",
    "MI": "myocardial infarction",
    "CVD": "cardiovascular disease",
    "CKD": "chronic kidney disease",
    "ESRD": "end-stage renal disease",
    "AML": "acute myeloid leukemia",
    "ALL": "acute lymphoblastic leukemia",
    "CML": "chronic myeloid leukemia",
    "NHL": "non-Hodgkin lymphoma",
    "NSCLC": "non-small cell lung cancer",
    "SCLC": "small cell lung cancer",
    "HCC": "hepatocellular carcinoma",
    "RCC": "renal cell carcinoma",
    "CRC": "colorectal cancer",
    "ASD": "autism spectrum disorder",
    "ADHD": "attention deficit hyperactivity disorder",
    "MDD": "major depressive disorder",
    "OCD": "obsessive-compulsive disorder",
    "PTSD": "post-traumatic stress disorder",
    "RA": "rheumatoid arthritis",
    "SLE": "systemic lupus erythematosus",
    "IBD": "inflammatory bowel disease",
    "UC": "ulcerative colitis",
    "CF": "cystic fibrosis",
    "DMD": "Duchenne muscular dystrophy",
    "SMA": "spinal muscular atrophy",
    "MFS": "Marfan syndrome",
    "NDI": "nephrogenic diabetes insipidus",
    "DI": "diabetes insipidus",
    "T2D": "type 2 diabetes",
    "T1D": "type 1 diabetes",
    "DKA": "diabetic ketoacidosis",
    "TDF": "tenofovir disoproxil fumarate",
    "LAM": "lamivudine",
    "AZT": "zidovudine",
    "EFV": "efavirenz",
    "3TC": "lamivudine",
    "ABC": "abacavir",
    "FTC": "emtricitabine",
    "TNF": "tumor necrosis factor",
    "IL-6": "interleukin-6",
    "IL-1": "interleukin-1",
    "IFN": "interferon",
    "VEGF": "vascular endothelial growth factor",
    "EGF": "epidermal growth factor",
    "EGFR": "epidermal growth factor receptor",
    "ACE": "angiotensin-converting enzyme",
    "GABA": "gamma-aminobutyric acid",
    "NO": "nitric oxide",
    "ROS": "reactive oxygen species",
    "PCR": "polymerase chain reaction",
    "SNP": "single nucleotide polymorphism",
    "CNS": "central nervous system",
    "BBB": "blood-brain barrier",
    "GI": "gastrointestinal",
    "BMI": "body mass index",
    "LQTS": "long QT syndrome",
    "AF": "atrial fibrillation",
    "VT": "ventricular tachycardia",
    "SVT": "supraventricular tachycardia",
    "PE": "pulmonary embolism",
    "DVT": "deep vein thrombosis",
    "TIA": "transient ischemic attack",
    "CAD": "coronary artery disease",
    "PAD": "peripheral artery disease",
    "NAFLD": "non-alcoholic fatty liver disease",
    "NASH": "non-alcoholic steatohepatitis",
    "IBS": "irritable bowel syndrome",
    "GERD": "gastroesophageal reflux disease",
    "UTI": "urinary tract infection",
    "ARDS": "acute respiratory distress syndrome",
    "DIC": "disseminated intravascular coagulation",
    "HUS": "hemolytic uremic syndrome",
    "TTP": "thrombotic thrombocytopenic purpura",
    "GBS": "Guillain-Barre syndrome",
    "MG": "myasthenia gravis",
    "PKD": "polycystic kidney disease",
    "WPW": "Wolff-Parkinson-White syndrome",
    "VSD": "ventricular septal defect",
    "ASD_cardiac": "atrial septal defect",
    "PDA": "patent ductus arteriosus",
    "TOF": "tetralogy of Fallot",
    "HLHS": "hypoplastic left heart syndrome",
    "NF1": "neurofibromatosis type 1",
    "NF2": "neurofibromatosis type 2",
    "TSC": "tuberous sclerosis complex",
    "VHL": "von Hippel-Lindau disease",
    "FAP": "familial adenomatous polyposis",
    "BRCA": "breast cancer gene",
    "TTX": "tetrodotoxin",
    "ATP": "adenosine triphosphate",
    "ADP": "adenosine diphosphate",
    "cAMP": "cyclic adenosine monophosphate",
    "NAD": "nicotinamide adenine dinucleotide",
    "NADP": "nicotinamide adenine dinucleotide phosphate",
    "GSH": "glutathione",
    "5-HT": "serotonin",
    "DA": "dopamine",
    "NE": "norepinephrine",
    "Ach": "acetylcholine",
    "DOPA": "dihydroxyphenylalanine",
    "MTX": "methotrexate",
    "DOX": "doxorubicin",
    "CIS": "cisplatin",
    "5-FU": "fluorouracil",
}

# Common biomedical abbreviations that are ambiguous
AMBIGUOUS_TERMS = {
    "AD": [
        {"term": "Alzheimer's disease", "context_words": [
            "neuron", "brain", "amyloid", "tau", "cognitive", "dementia",
            "hippocamp", "cortex", "plaque", "tangle", "APOE", "APP",
        ]},
        {"term": "atopic dermatitis", "context_words": [
            "skin", "eczema", "itch", "dermat", "epiderm", "topical",
            "allergic", "IgE", "rash",
        ]},
        {"term": "autosomal dominant", "context_words": [
            "inherit", "mutation", "pedigree", "penetran", "allele",
        ]},
    ],
    "PD": [
        {"term": "Parkinson's disease", "context_words": [
            "dopamin", "substantia nigra", "motor", "tremor", "SNCA",
            "alpha-synuclein", "lewy", "basal ganglia",
        ]},
        {"term": "peritoneal dialysis", "context_words": [
            "dialysis", "renal", "kidney", "periton", "fluid",
        ]},
    ],
    "MS": [
        {"term": "multiple sclerosis", "context_words": [
            "myelin", "demyelinat", "relapsing", "lesion", "MRI",
            "oligodendrocyte", "spinal cord",
        ]},
        {"term": "mass spectrometry", "context_words": [
            "proteom", "spectrum", "m/z", "peptide", "LC-MS",
        ]},
    ],
    "ALS": [
        {"term": "amyotrophic lateral sclerosis", "context_words": [
            "motor neuron", "SOD1", "C9orf72", "TDP-43", "FUS",
            "bulbar", "spinal",
        ]},
    ],
}


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity (0-1)."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    distance = matrix[len1][len2]
    return 1.0 - distance / max(len1, len2)


def _trigram_similarity(s1: str, s2: str) -> float:
    """Compute trigram-based Jaccard similarity."""
    def trigrams(s):
        s = f"  {s.lower()}  "
        return {s[i:i+3] for i in range(len(s) - 2)}

    t1, t2 = trigrams(s1), trigrams(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


_DISEASE_PREFIXES = re.compile(
    r'^(congenital|familial|chronic|acute|severe|mild|moderate|inherited|'
    r'autosomal dominant|autosomal recessive|hereditary|idiopathic|'
    r'progressive|recurrent|benign|malignant|primary|secondary)\s+',
    re.IGNORECASE,
)


def _clean_text(text: str) -> list[str]:
    """Generate normalized text variants for lookup.

    Returns list of variants to try, in priority order.
    """
    variants = [text]

    # Replace colons with spaces: "NAD(P)H:quinone" -> "NAD(P)H quinone"
    no_colon = text.replace(":", " ")
    if no_colon != text:
        variants.append(no_colon)

    # Strip parenthetical content: "Na(v)1.5" -> "Nav1.5", "Na1.5"
    no_parens = re.sub(r'\(([^)]*)\)', r'\1', text)
    if no_parens != text:
        variants.append(no_parens)
        # Also try colon-stripped + parentheses-expanded
        no_parens_no_colon = no_parens.replace(":", " ")
        if no_parens_no_colon != no_parens:
            variants.append(no_parens_no_colon)
    stripped_parens = re.sub(r'\([^)]*\)', '', text).strip()
    if stripped_parens and stripped_parens != text:
        variants.append(stripped_parens)

    # Strip common disease/condition prefixes
    stripped_prefix = _DISEASE_PREFIXES.sub('', text).strip()
    if stripped_prefix and stripped_prefix.lower() != text.lower():
        variants.append(stripped_prefix)

    # Depluralize: "anthracyclines" -> "anthracycline", "deficiencies" -> "deficiency"
    for v in list(variants):
        if v.endswith('ies'):
            variants.append(v[:-3] + 'y')
        elif v.endswith('ses') or v.endswith('xes') or v.endswith('zes'):
            variants.append(v[:-2])
        elif v.endswith('s') and not v.endswith('ss'):
            variants.append(v[:-1])

    # Remove seen duplicates while preserving order
    seen = set()
    unique = []
    for v in variants:
        if v.lower() not in seen and v.strip():
            seen.add(v.lower())
            unique.append(v)
    return unique


# Common receptor/protein name patterns -> gene symbol patterns
# These handle cases where text says "beta-1 adrenoceptor" but dictionary has "ADRB1"
RECEPTOR_PATTERNS = [
    # "beta-1 adrenoceptor" or "beta-1 adrenergic receptor" -> "ADRB1"
    (r'beta[- ]?(\d)\s*(?:adrenoceptor|adrenergic receptor)', r'ADRB\1'),
    # "alpha-1 adrenoceptor" -> "ADRA1A" (approximate)
    (r'alpha[- ]?(\d)\s*(?:adrenoceptor|adrenergic receptor)', r'ADRA\1'),
    # "mu opioid receptor" -> "OPRM1"
    (r'mu[- ]?opioid receptor', 'OPRM1'),
    # "kappa opioid receptor" -> "OPRK1"
    (r'kappa[- ]?opioid receptor', 'OPRK1'),
    # "delta opioid receptor" -> "OPRD1"
    (r'delta[- ]?opioid receptor', 'OPRD1'),
    # "dopamine D2 receptor" -> "DRD2"
    (r'dopamine\s*D(\d)\s*receptor', r'DRD\1'),
    # "serotonin 5-HT2A receptor" -> "HTR2A"
    (r'(?:serotonin\s*)?5[- ]?HT(\d)([A-C])?\s*receptor', r'HTR\1\2'),
    # "beta-2 receptor" (short form) -> "ADRB2"
    (r'beta[- ]?(\d)\s*receptor', r'ADRB\1'),
]


def disambiguate_abbreviation(term: str, context_text: str) -> str | None:
    """Resolve an ambiguous abbreviation using surrounding text context."""
    candidates = AMBIGUOUS_TERMS.get(term.upper())
    if not candidates:
        return None

    context_lower = context_text.lower()
    scores = []
    for candidate in candidates:
        score = sum(1 for word in candidate["context_words"] if word.lower() in context_lower)
        scores.append((score, candidate["term"]))

    scores.sort(reverse=True)
    if scores[0][0] > 0:
        return scores[0][1]
    return None


# Pattern for "X gene Y" or "X protein Y" where Y is a gene symbol
# Use a word boundary or sentence-level anchor to avoid grabbing too much context
_GENE_DEFINITION_RE = re.compile(
    r'(?:^|[.,;]\s+|\b(?:the|a|an)\s+)([A-Za-z][a-zA-Z\s\-\(\):0-9]{3,80}?)\s+(?:gene|protein|enzyme)\s+([A-Z][A-Z0-9]{1,10})\b'
)


def _extract_abbreviation_map(text: str) -> dict[str, str]:
    """Extract abbreviation -> full name mappings from text.

    Finds patterns like:
    - "long QT syndrome (LQTS)" -> {"LQTS": "long QT syndrome"}
    - "carbonyl reductase 3 gene CBR3" -> {"CBR3": "carbonyl reductase 3"}
    Also handles nested parentheses like "NAD(P)H:quinone oxidoreductase 1 (NQO1)".
    """
    abbrev_map = {}
    for m in _ABBREV_DEFINITION_RE.finditer(text):
        full_name = m.group(1).strip()
        abbrev = m.group(2).strip()
        # Skip if "full name" is actually just another abbreviation
        if full_name.isupper() and len(full_name) <= 10:
            continue
        abbrev_map[abbrev] = full_name

    # Also find "X gene Y" patterns
    for m in _GENE_DEFINITION_RE.finditer(text):
        full_name = m.group(1).strip()
        symbol = m.group(2).strip()
        if full_name.isupper() and len(full_name) <= 10:
            continue
        # Strip leading articles and conjunctions
        full_name = re.sub(r'^(?:and\s+)?(?:the|a|an)\s+', '', full_name, flags=re.IGNORECASE).strip()
        if symbol not in abbrev_map:
            abbrev_map[symbol] = full_name

    return abbrev_map


class EntityNormalizer:
    """Normalize extracted entity mentions to canonical IDs."""

    def __init__(self, dictionary: DictionaryLookup | None = None):
        self.dictionary = dictionary or DictionaryLookup()

    def _search_with_fallback(
        self,
        text: str,
        entity_type: str | None,
        method: str = "exact",
    ) -> list[DictMatch]:
        """Search dictionary with type filter, then cross-type, then untyped.

        This handles cases like DISEASE entities stored as PHENOTYPE in HPO,
        or GENE entities stored as PROTEIN.
        """
        # 1. Try with exact type
        if method == "exact":
            matches = self.dictionary.exact_match(text, entity_type)
        else:
            matches = self.dictionary.search(text, entity_type, limit=5)
        if matches:
            return matches

        # 2. Try compatible types (e.g., DISEASE -> also search PHENOTYPE)
        if entity_type:
            normalized = _normalize_type(entity_type)
            compatible = COMPATIBLE_TYPES.get(normalized or entity_type.upper(), [])
            for compat_type in compatible:
                if compat_type != normalized:
                    if method == "exact":
                        matches = self.dictionary.exact_match(text, compat_type)
                    else:
                        matches = self.dictionary.search(text, compat_type, limit=5)
                    if matches:
                        return matches

        # 3. Try without any type filter, but filter out completely incompatible types
        if entity_type:
            if method == "exact":
                matches = self.dictionary.exact_match(text, None)
            else:
                matches = self.dictionary.search(text, None, limit=5)
            if matches:
                # For short terms (abbreviations), require at least some type compatibility
                # to avoid "MS" (DISEASE) matching gene:MTR
                if len(text) <= 4 and entity_type:
                    normalized = _normalize_type(entity_type)
                    compat = set(COMPATIBLE_TYPES.get(normalized or entity_type.upper(), [normalized or entity_type.upper()]))
                    filtered = [m for m in matches if _normalize_type(m.entity_type) in compat]
                    if filtered:
                        return filtered
                    # If no compatible matches for short terms, return empty
                    return []
                return matches

        return []

    def normalize_entity(
        self,
        entity: ExtractedEntity,
        context_text: str = "",
    ) -> ExtractedEntity:
        """Normalize a single entity mention.

        Tries in order:
        1. Exact match (with type fallback)
        2. Common abbreviation expansion (unambiguous: CHF, HBV, etc.)
        3. Abbreviation disambiguation (ambiguous: AD, PD, MS)
        4. Inline abbreviation expansion (from context text)
        5. Receptor/protein name patterns -> gene symbols
        6. Text variants (plurals, parentheses)
        7. FTS search (with type fallback)
        8. Suffix stripping
        """
        if not self.dictionary.is_available():
            return entity

        text = entity.text.strip()

        # 1. Exact match with type fallback
        matches = self._search_with_fallback(text, entity.type, "exact")
        if matches:
            best = self._rank_candidates(matches, text, entity.type)
            entity.canonical_id = best.canonical_id
            entity.canonical_name = best.name
            entity.confidence = min(entity.confidence, best.score)
            return entity

        # 2. Common abbreviation expansion (unambiguous)
        if text.upper() in COMMON_ABBREVIATIONS:
            expanded = COMMON_ABBREVIATIONS[text.upper()]
            matches = self._search_with_fallback(expanded, entity.type, "exact")
            if not matches:
                for variant in _clean_text(expanded)[1:]:
                    matches = self._search_with_fallback(variant, entity.type, "exact")
                    if matches:
                        break
            if not matches:
                matches = self._search_with_fallback(expanded, entity.type, "fts")
            if matches:
                best = self._rank_candidates(matches, expanded, entity.type)
                entity.canonical_id = best.canonical_id
                entity.canonical_name = best.name
                entity.confidence = min(entity.confidence, 0.85)
                return entity

        # 3. Abbreviation disambiguation (ambiguous terms)
        if text.upper() in AMBIGUOUS_TERMS and context_text:
            resolved = disambiguate_abbreviation(text, context_text)
            if resolved:
                matches = self._search_with_fallback(resolved, entity.type, "exact")
                if matches:
                    best = self._rank_candidates(matches, resolved, entity.type)
                    entity.canonical_id = best.canonical_id
                    entity.canonical_name = best.name
                    entity.confidence = min(entity.confidence, 0.8)
                    return entity

        # 3. Inline abbreviation expansion: "LQTS" -> "long QT syndrome" from context
        if text.isupper() and len(text) <= 10 and context_text:
            abbrev_map = _extract_abbreviation_map(context_text)
            expanded = abbrev_map.get(text)
            if expanded:
                # Try exact match on expanded form and its variants
                for variant in _clean_text(expanded):
                    matches = self._search_with_fallback(variant, entity.type, "exact")
                    if matches:
                        best = self._rank_candidates(matches, expanded, entity.type)
                        entity.canonical_id = best.canonical_id
                        entity.canonical_name = best.name
                        entity.confidence = min(entity.confidence, 0.85)
                        return entity
                # Try FTS on expanded form
                matches = self._search_with_fallback(expanded, entity.type, "fts")
                if matches:
                    best = self._rank_candidates(matches, expanded, entity.type)
                    if best.score >= 0.6:
                        entity.canonical_id = best.canonical_id
                        entity.canonical_name = best.name
                        entity.confidence = min(entity.confidence, 0.8)
                        return entity

        # 4. Receptor/protein name patterns -> gene symbols
        for pattern, replacement in RECEPTOR_PATTERNS:
            m = re.match(pattern, text, re.IGNORECASE)
            if m:
                gene_symbol = m.expand(replacement)
                matches = self._search_with_fallback(gene_symbol, "GENE", "exact")
                if matches:
                    best = self._rank_candidates(matches, gene_symbol, entity.type)
                    entity.canonical_id = best.canonical_id
                    entity.canonical_name = best.name
                    entity.confidence = min(entity.confidence, 0.85)
                    return entity

        # 5. Text variants (plurals, parentheses stripped)
        for variant in _clean_text(text)[1:]:  # skip original, already tried
            matches = self._search_with_fallback(variant, entity.type, "exact")
            if matches:
                best = self._rank_candidates(matches, text, entity.type)
                entity.canonical_id = best.canonical_id
                entity.canonical_name = best.name
                entity.confidence = min(entity.confidence, best.score * 0.95)
                return entity

        # 6. FTS search with type fallback
        for variant in _clean_text(text):
            matches = self._search_with_fallback(variant, entity.type, "fts")
            if matches:
                best = self._rank_candidates(matches, text, entity.type)
                if best.score >= 0.5:
                    entity.canonical_id = best.canonical_id
                    entity.canonical_name = best.name
                    entity.confidence = min(entity.confidence, best.score)
                    return entity

        # 7. Suffix stripping — remove common biomedical suffixes
        simplified = re.sub(
            r'\s*(gene|protein|receptor|inhibitor|disease|syndrome|deficiency|deficiencies)s?\s*$',
            '', text, flags=re.IGNORECASE
        ).strip()
        if simplified and simplified != text:
            matches = self._search_with_fallback(simplified, entity.type, "fts")
            if matches:
                best = self._rank_candidates(matches, text, entity.type)
                if best.score >= 0.5:
                    entity.canonical_id = best.canonical_id
                    entity.canonical_name = best.name
                    entity.confidence = min(entity.confidence, best.score * 0.9)
                    return entity

        # 8. Reverse abbreviation lookup: if context has "full name (ABBR)" or
        # "full name gene ABBR", and this entity IS the full name (or a variant),
        # try normalizing the abbreviation instead.
        if context_text and len(text) > 10:
            abbrev_map = _extract_abbreviation_map(context_text)
            for abbr, full_name in abbrev_map.items():
                # Check containment OR significant word overlap
                text_l = text.lower()
                full_l = full_name.lower()
                if full_l in text_l or text_l in full_l:
                    match = True
                else:
                    # Word overlap: if most content words match, it's the same entity
                    # "NAD(P)H:quinone oxidoreductase 1" vs
                    # "nicotinamide adenine dinucleotide phosphate:quinone oxidoreductase 1"
                    text_words = set(re.split(r'[\s:,\-\(\)]+', text_l)) - {'', 'the', 'a', 'and', 'of'}
                    full_words = set(re.split(r'[\s:,\-\(\)]+', full_l)) - {'', 'the', 'a', 'and', 'of'}
                    overlap = len(text_words & full_words)
                    min_words = min(len(text_words), len(full_words))
                    match = min_words > 0 and overlap / min_words >= 0.5
                if match:
                    matches = self._search_with_fallback(abbr, entity.type, "exact")
                    if matches:
                        best = self._rank_candidates(matches, abbr, entity.type)
                        entity.canonical_id = best.canonical_id
                        entity.canonical_name = best.name
                        entity.confidence = min(entity.confidence, 0.8)
                        return entity

        return entity

    def _rank_candidates(
        self,
        candidates: list[DictMatch],
        query: str,
        expected_type: str | None = None,
    ) -> DictMatch:
        """Rank candidates by string similarity and type consistency."""
        normalized_expected = _normalize_type(expected_type) if expected_type else None
        # Also get compatible types for bonus scoring
        compat_types = set()
        if normalized_expected:
            compat_types = set(COMPATIBLE_TYPES.get(normalized_expected, [normalized_expected]))

        scored = []
        for c in candidates:
            # String similarity against canonical name
            lev = _levenshtein_ratio(query, c.name)
            tri = _trigram_similarity(query, c.name)
            str_score = max(lev, tri)

            # Case-sensitive and synonym-similarity bonus.
            # "dGK" should prefer DGUOK (synonym "dGK" = exact case match)
            # over DGKB (synonym "DGK" = case-insensitive only).
            case_bonus = 0.0
            synonym_str_bonus = 0.0
            if c.match_type in ("exact", "synonym") and self.dictionary:
                try:
                    conn = self.dictionary._get_conn()
                    syns = conn.execute(
                        "SELECT synonym FROM synonyms WHERE canonical_id = ?",
                        (c.canonical_id,),
                    ).fetchall()
                    for syn in syns:
                        syn_text = syn["synonym"]
                        # Case-exact match is a very strong signal
                        if syn_text == query:
                            case_bonus = 0.25
                            synonym_str_bonus = max(synonym_str_bonus, 1.0 - str_score)
                            break
                        # Also boost string similarity against synonym
                        syn_sim = max(_levenshtein_ratio(query, syn_text),
                                      _trigram_similarity(query, syn_text))
                        synonym_str_bonus = max(synonym_str_bonus, syn_sim - str_score)
                    if c.name == query:
                        case_bonus = max(case_bonus, 0.25)
                except Exception:
                    pass
            # Apply synonym string bonus (difference between synonym match and name match)
            # This ensures candidates matched via a better synonym get credit
            if synonym_str_bonus > 0:
                str_score += synonym_str_bonus

            # For FTS/fuzzy matches, also check synonyms — the match may have
            # been via a synonym that better matches the query than the canonical name
            if c.match_type in ("fts", "fuzzy") and str_score < 0.6 and self.dictionary:
                try:
                    conn = self.dictionary._get_conn()
                    syns = conn.execute(
                        "SELECT synonym FROM synonyms WHERE canonical_id = ?",
                        (c.canonical_id,),
                    ).fetchall()
                    for syn in syns:
                        syn_lev = _levenshtein_ratio(query, syn["synonym"])
                        syn_tri = _trigram_similarity(query, syn["synonym"])
                        str_score = max(str_score, syn_lev, syn_tri)
                        if str_score >= 0.8:
                            break  # Good enough
                except Exception:
                    pass

            # Type consistency bonus — compatible types also get partial bonus
            type_bonus = 0.0
            if compat_types:
                if c.entity_type in compat_types:
                    type_bonus = 0.1
                elif _normalize_type(c.entity_type) in compat_types:
                    type_bonus = 0.08

            # Match type bonus
            match_bonus = {"exact": 0.2, "synonym": 0.15, "fts": 0.05, "fuzzy": 0.0}.get(c.match_type, 0.0)

            total = min(str_score, 1.0) + type_bonus + match_bonus + case_bonus
            scored.append((total, c))

        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_match = scored[0]
        best_match.score = min(1.0, best_score)
        return best_match

    def normalize_result(self, result: ExtractionResult) -> ExtractionResult:
        """Normalize all entities in an extraction result."""
        for entity in result.entities:
            self.normalize_entity(entity, context_text=result.text)
        return result
