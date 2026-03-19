"""Rule-based entity and relationship extraction using dictionary + verb patterns.

No LLM calls. Uses:
1. Dictionary scanning for entity recognition
2. Sentence-level co-occurrence for candidate pairs
3. Verb/keyword pattern matching for relationship classification
4. Negation detection
5. Entity type constraints for biological plausibility
"""

import re
from dataclasses import dataclass, field

from ..dictionaries.lookup import DictionaryLookup, DictMatch
from ..normalize import (
    COMMON_ABBREVIATIONS,
    AMBIGUOUS_TERMS,
    COMPATIBLE_TYPES,
    _clean_text,
    disambiguate_abbreviation,
)
from ..schema import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    RelationshipContext,
)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'       # period/excl/question + space + uppercase
    r'|(?<=\.\))\s+(?=[A-Z])'        # ".) " boundary
    r'|(?<=[.!?])\s*\n'              # period + newline
)


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into (sentence, start, end) tuples."""
    parts = _SENT_RE.split(text)
    sentences = []
    offset = 0
    for part in parts:
        start = text.find(part, offset)
        if start == -1:
            start = offset
        end = start + len(part)
        sentences.append((part.strip(), start, end))
        offset = end
    return [(s, st, en) for s, st, en in sentences if s]


# ---------------------------------------------------------------------------
# Verb / keyword pattern definitions
# ---------------------------------------------------------------------------

@dataclass
class RelPattern:
    """A keyword pattern that signals a relationship type."""
    keywords: list[str]
    rel_type: str
    direction: str
    # If set, only match when subject/object types are in these sets
    subject_types: set[str] = field(default_factory=set)
    object_types: set[str] = field(default_factory=set)


# Patterns ordered from most specific to most general
RELATIONSHIP_PATTERNS: list[RelPattern] = [
    # Treatment / therapeutic
    RelPattern(
        keywords=["treats", "treated", "treatment of", "treatment for",
                  "therapy for", "therapeutic", "controlled",
                  "responded to", "response to", "effective against",
                  "ameliorates", "alleviates", "attenuates",
                  "prevents", "prevented", "prophylaxis"],
        rel_type="downregulates",
        direction="negative",
    ),
    # Causation / induction
    RelPattern(
        keywords=["causes", "caused", "induces", "induced",
                  "leads to", "results in", "triggers", "elicits",
                  "produces", "contributes to"],
        rel_type="causes",
        direction="positive",
    ),
    # Upregulation / increase
    RelPattern(
        keywords=["increases", "increased", "upregulates", "upregulated",
                  "activates", "activated", "enhances", "enhanced",
                  "promotes", "promoted", "stimulates", "stimulated",
                  "elevates", "elevated", "amplifies", "potentiates"],
        rel_type="upregulates",
        direction="positive",
    ),
    # Downregulation / decrease
    RelPattern(
        keywords=["decreases", "decreased", "downregulates", "downregulated",
                  "reduces", "reduced", "suppresses", "suppressed",
                  "diminishes", "diminished", "lowers", "lowered",
                  "attenuated", "mitigated"],
        rel_type="downregulates",
        direction="negative",
    ),
    # Inhibition
    RelPattern(
        keywords=["inhibits", "inhibited", "inhibitor of", "inhibition of",
                  "blocks", "blocked", "antagonist", "antagonizes",
                  "suppresses", "represses", "repressed"],
        rel_type="inhibits",
        direction="negative",
    ),
    # Binding
    RelPattern(
        keywords=["binds", "binding", "bound to", "interacts with",
                  "affinity for", "ligand", "substrate",
                  "receptor for"],
        rel_type="binds",
        direction="neutral",
    ),
    # Risk / predisposition
    RelPattern(
        keywords=["risk factor", "risk of", "predisposition",
                  "susceptibility", "predispose", "predisposed",
                  "increases risk", "increased risk"],
        rel_type="increases_risk",
        direction="positive",
    ),
    # Compound adjective patterns: "X-related Y", "X-induced Y", "X-associated Y"
    RelPattern(
        keywords=["-related", "-induced", "-associated", "-mediated",
                  "-linked", "-dependent"],
        rel_type="associated_with",
        direction="neutral",
    ),
    # General association
    RelPattern(
        keywords=["associated with", "association between", "correlated with",
                  "correlation between", "linked to", "implicated in",
                  "involved in", "role in", "contributes to",
                  "responsible for", "attributed to", "related to"],
        rel_type="associated_with",
        direction="neutral",
    ),
    # Regulation (generic)
    RelPattern(
        keywords=["regulates", "regulated", "regulation of",
                  "modulates", "modulated", "modulation of",
                  "controls", "controlled"],
        rel_type="regulates",
        direction="neutral",
    ),
    # Metabolism / conversion
    RelPattern(
        keywords=["metabolized", "metabolite of", "converted to",
                  "conversion of", "biotransformation", "metabolizes"],
        rel_type="interacts_with",
        direction="neutral",
    ),
    # Expression
    RelPattern(
        keywords=["expressed in", "expression of", "expression in",
                  "overexpressed", "underexpressed"],
        rel_type="expressed_in",
        direction="neutral",
    ),
]

# Negation cues
NEGATION_CUES = [
    "not", "no", "nor", "neither", "without", "absence of",
    "failed to", "did not", "does not", "do not", "cannot",
    "unlikely", "no evidence", "no association", "no significant",
    "lack of", "non-significant",
]

# Biologically plausible entity pair types for relationships
PLAUSIBLE_PAIRS: set[tuple[str, str]] = {
    ("CHEMICAL", "DISEASE"),
    ("CHEMICAL", "GENE"),
    ("CHEMICAL", "CHEMICAL"),
    ("GENE", "DISEASE"),
    ("GENE", "GENE"),
    ("VARIANT", "DISEASE"),
    ("VARIANT", "GENE"),
    ("GENE", "CELL_TYPE"),
    ("GENE", "ORGANISM"),
    ("CHEMICAL", "ORGANISM"),
    ("CHEMICAL", "VARIANT"),
}


def _is_plausible_pair(type_a: str, type_b: str) -> bool:
    """Check if two entity types can have a biologically plausible relationship."""
    a, b = type_a.upper(), type_b.upper()
    return (a, b) in PLAUSIBLE_PAIRS or (b, a) in PLAUSIBLE_PAIRS


# ---------------------------------------------------------------------------
# Dictionary-based entity scanning
# ---------------------------------------------------------------------------

# Words/phrases that are valid dictionary entries but too generic to extract.
# Includes study-design terms, methodology, statistics, and molecular biology
# jargon that MeSH classifies as chemicals/processes but aren't real entities.
_STOPWORDS = {
    # Function words
    "a", "an", "the", "in", "on", "at", "to", "of", "is", "are", "was",
    "were", "be", "been", "or", "and", "not", "no", "as", "by", "for",
    "with", "from", "that", "this", "it", "its", "has", "had", "have",
    "we", "our", "they", "their", "but", "if", "so", "than", "then",
    "also", "may", "can", "will", "each", "all", "both", "more", "most",
    # Units / generic
    "new", "two", "six", "one", "day", "per", "mg", "ml", "hr", "vs",
    # Study design / methodology
    "type", "gene", "protein", "cell", "cells", "study", "studies",
    "data", "results", "group", "groups", "effect", "effects",
    "analysis", "test", "case", "cases", "control", "controls",
    "role", "dose", "doses", "level", "levels", "rate", "rates",
    "time", "form", "forms", "method", "methods",
    "risk factors", "risk factor", "case-control studies",
    "nested case-control study", "case-control study",
    "cohort study", "cohort studies", "clinical trial", "clinical trials",
    "odds ratio", "odds ratios", "confidence interval",
    "prevalence", "incidence", "mortality", "morbidity",
    "prognosis", "diagnosis", "treatment", "therapy",
    "screening", "assay", "assays", "sample", "samples",
    "after", "before", "during", "between", "only", "first",
    "found", "used", "observed", "reported", "identified", "detected",
    # Demographics / population
    "patient", "patients", "human", "humans", "subjects", "families",
    "newborn", "infant", "child", "children", "adult", "adults",
    "male", "female", "man", "woman", "men", "women", "boy", "girl",
    "age", "sex", "gender", "weight",
    # Genetics jargon
    "mutation", "mutations", "polymorphism", "polymorphisms",
    "substitution", "allele", "alleles", "genotype", "genotypes",
    "phenotype", "phenotypes", "haplotype", "haplotypes",
    "allele frequency", "gene frequency",
    "expression", "function", "activity", "domain",
    "sequence", "sequencing", "DNA", "RNA", "mRNA", "cDNA",
    "heterozygous", "homozygous", "wild-type", "recombinant",
    "codon", "codons", "exon", "exons", "intron", "introns",
    "isoform", "isoforms", "amino acid", "amino acids",
    # Body fluids / materials (not entities)
    "serum", "plasma", "blood", "urine", "tissue", "tissues",
    # PK / generic biology
    "pharmacokinetics", "bioavailability", "absorption",
    "metabolism", "excretion", "clearance", "half-life",
    "metabolizers", "extensive metabolizers", "poor metabolizers",
    "body mass index",
    # Experimental techniques (not entities)
    "site-directed mutagenesis", "DNA sequencing", "western blotting",
    "electrophoresis", "chromatography", "spectroscopy", "microscopy",
    "actinomycin D", "actinomycin",  # experimental reagent, not therapeutic
    "multivariate analyses", "multivariate analysis",
    # MeSH terms that are not biomedical entities
    "japan", "japanese", "china", "chinese", "korea", "korean",
    "european", "caucasian", "caucasians", "african", "asian",
    "administration", "organization and administration",
    "5'-untranslated region", "untranslated region", "untranslated regions",
    "3'-untranslated region",
    "trinucleotide repeat", "trinucleotide repeats",
    "chromosome", "chromosomes", "autosomal",
    "skin biopsy", "biopsy",
    "intraperitoneal", "subcutaneous", "intravenous",
    "placebo", "vector",
    # Generic words that match dictionary entries
    "who", "plays", "coding", "foundation", "autopsy", "impact",
    "drugs", "other", "position", "association", "associations",
    "frequency", "imbalance", "part of", "parents", "parent",
    "sister", "brother", "family", "families", "survivors", "survivor",
    "disease", "diseases", "recurrence", "disorder", "disorders",
    "syndrome", "syndromes", "condition", "conditions",
    "sodium channel", "ion channel", "calcium channel", "potassium channel",
    "oxidoreductase",
    # Amino acids when used in variant descriptions
    "valine", "methionine", "alanine", "isoleucine", "leucine",
    "proline", "threonine", "serine", "glycine", "cysteine",
    "tyrosine", "tryptophan", "phenylalanine", "histidine",
    "lysine", "arginine", "aspartate", "glutamate", "asparagine", "glutamine",
    # Nucleosides in variant descriptions
    "adenosine", "guanosine", "cytidine", "thymidine",
    # Roman numerals that match chemical/gene entries
    "I", "II", "III", "IV", "V", "VI",
}

# Entity types from dictionary that we trust for relationship extraction
_ENTITY_TYPES_FOR_RELS = {"GENE", "DISEASE", "CHEMICAL", "VARIANT", "ORGANISM", "CELL_TYPE"}

# Types from the dictionary that are too broad for pattern matching
_SKIP_TYPES = {"BIOLOGICAL_PROCESS", "MOLECULAR_FUNCTION", "PATHWAY", "ANATOMICAL_STRUCTURE"}


def _scan_entities_in_text(
    text: str,
    dictionary: DictionaryLookup,
    max_ngram: int = 6,
) -> list[ExtractedEntity]:
    """Scan text for dictionary entities using sliding n-gram window."""
    words = text.split()
    found: dict[str, ExtractedEntity] = {}  # canonical_id -> entity
    # Track matched text spans to prefer longer matches
    matched_spans: dict[str, str] = {}  # text_lower -> canonical_id

    for n in range(max_ngram, 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            # Strip trailing punctuation
            cleaned = phrase.rstrip(".,;:!?()[]")
            if len(cleaned) < 3:
                continue
            if cleaned.lower() in _STOPWORDS or cleaned in _STOPWORDS:
                continue

            matches = dictionary.exact_match(cleaned)
            if not matches:
                # Try common abbreviation expansion
                upper = cleaned.upper()
                if upper in COMMON_ABBREVIATIONS:
                    expanded = COMMON_ABBREVIATIONS[upper]
                    matches = dictionary.exact_match(expanded)

            for m in matches:
                if m.entity_type in _SKIP_TYPES:
                    continue
                # Skip if this text is a substring of an already-matched longer phrase
                cl = cleaned.lower()
                already_covered = False
                for existing_text in matched_spans:
                    if cl in existing_text and cl != existing_text:
                        already_covered = True
                        break
                if already_covered:
                    continue

                if m.canonical_id not in found:
                    # Find character offset
                    start = text.find(cleaned)
                    if start == -1:
                        start = 0
                    found[m.canonical_id] = ExtractedEntity(
                        text=cleaned,
                        type=m.entity_type,
                        start=start,
                        end=start + len(cleaned),
                        canonical_id=m.canonical_id,
                        canonical_name=m.name,
                        confidence=0.90,
                    )
                    matched_spans[cl] = m.canonical_id

    return list(found.values())


# ---------------------------------------------------------------------------
# Relationship pattern matching
# ---------------------------------------------------------------------------

def _check_negation(text: str, start: int, end: int) -> bool:
    """Check if there's a negation cue near the keyword match."""
    # Look in a window around the match
    window_start = max(0, start - 40)
    window_end = min(len(text), end + 20)
    window = text[window_start:window_end].lower()
    return any(cue in window for cue in NEGATION_CUES)


def _extract_compound_adjective_rels(
    sentence: str,
    entities_in_sent: list[ExtractedEntity],
) -> list[ExtractedRelationship]:
    """Find 'X-related Y' patterns in compound adjectives."""
    results = []
    # Pattern: entity_text-keyword other_entity
    compound_re = re.compile(
        r'(\w[\w\s]*?)[-\s](?:related|induced|associated|mediated|linked|dependent)'
        r'\s+(\w[\w\s]*?)(?:[,.\s]|$)',
        re.IGNORECASE,
    )

    ent_texts = {e.text.lower(): e for e in entities_in_sent}

    for m in compound_re.finditer(sentence):
        prefix = m.group(1).strip().lower()
        suffix = m.group(2).strip().lower()

        subj = None
        obj = None
        for et, e in ent_texts.items():
            if et in prefix or prefix in et:
                subj = e
            if et in suffix or suffix in et:
                obj = e

        if subj and obj and subj.text != obj.text:
            if _is_plausible_pair(subj.type, obj.type):
                results.append(ExtractedRelationship(
                    subject=subj.text,
                    predicate=m.group(0).strip(),
                    object=obj.text,
                    type="associated_with",
                    direction="neutral",
                    negated=False,
                    context=RelationshipContext(),
                    confidence=0.55,
                ))

    return results


def _match_patterns_in_sentence(
    sentence: str,
    entities_in_sent: list[ExtractedEntity],
) -> list[ExtractedRelationship]:
    """Match verb/keyword patterns between entity pairs in a sentence."""
    results = []
    sent_lower = sentence.lower()

    # Generate all plausible entity pairs
    for i, e1 in enumerate(entities_in_sent):
        for e2 in entities_in_sent[i + 1:]:
            if not _is_plausible_pair(e1.type, e2.type):
                continue
            if e1.text.lower() == e2.text.lower():
                continue

            # Check each pattern
            for pattern in RELATIONSHIP_PATTERNS:
                for kw in pattern.keywords:
                    kw_lower = kw.lower()
                    if kw_lower in sent_lower:
                        kw_start = sent_lower.find(kw_lower)
                        negated = _check_negation(
                            sent_lower, kw_start, kw_start + len(kw_lower)
                        )

                        # Determine subject/object order:
                        # If e1 appears before keyword and e2 after, e1 is subject
                        e1_pos = sent_lower.find(e1.text.lower())
                        e2_pos = sent_lower.find(e2.text.lower())

                        if e1_pos < kw_start < e2_pos:
                            subj, obj = e1, e2
                        elif e2_pos < kw_start < e1_pos:
                            subj, obj = e2, e1
                        else:
                            # Default: first mentioned is subject
                            if e1_pos <= e2_pos:
                                subj, obj = e1, e2
                            else:
                                subj, obj = e2, e1

                        results.append(ExtractedRelationship(
                            subject=subj.text,
                            predicate=kw,
                            object=obj.text,
                            type=pattern.rel_type,
                            direction=pattern.direction,
                            negated=negated,
                            context=RelationshipContext(),
                            confidence=0.55 if not negated else 0.40,
                        ))
                        break  # One match per pattern per pair
                else:
                    continue
                break  # Found a pattern match, move to next pair

    # Also check compound adjective patterns
    results.extend(_extract_compound_adjective_rels(sentence, entities_in_sent))

    return results


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_with_patterns(
    text: str,
    dictionary: DictionaryLookup | None = None,
    precomputed_entities: list[ExtractedEntity] | None = None,
) -> ExtractionResult:
    """Extract entities and relationships using rule-based pattern matching.

    Args:
        text: The biomedical abstract text.
        dictionary: Optional dictionary for entity scanning. If None and no
            precomputed_entities, returns empty result.
        precomputed_entities: If provided, skip dictionary scanning and use
            these entities for relationship extraction.

    Returns:
        ExtractionResult with pattern-matched entities and relationships.
    """
    # Step 1: Get entities
    if precomputed_entities is not None:
        entities = list(precomputed_entities)
    elif dictionary is not None and dictionary.is_available():
        entities = _scan_entities_in_text(text, dictionary)
    else:
        return ExtractionResult(text=text, extraction_method="pattern_matcher")

    if not entities:
        return ExtractionResult(text=text, extraction_method="pattern_matcher")

    # Step 2: Split into sentences
    sentences = _split_sentences(text)

    # Step 3: Map entities to sentences
    relationships: list[ExtractedRelationship] = []
    seen_pairs: set[tuple[str, str]] = set()

    for sent_text, sent_start, sent_end in sentences:
        # Find entities in this sentence (only types useful for relationships)
        sent_lower = sent_text.lower()
        ents_in_sent = [
            e for e in entities
            if e.text.lower() in sent_lower
            and e.type.upper() in _ENTITY_TYPES_FOR_RELS
        ]

        if len(ents_in_sent) < 2:
            continue

        # Extract relationships from this sentence
        sent_rels = _match_patterns_in_sentence(sent_text, ents_in_sent)

        for r in sent_rels:
            pair = (
                min(r.subject.lower(), r.object.lower()),
                max(r.subject.lower(), r.object.lower()),
            )
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                relationships.append(r)

    return ExtractionResult(
        text=text,
        entities=entities,
        relationships=relationships,
        extraction_method="pattern_matcher",
    )
