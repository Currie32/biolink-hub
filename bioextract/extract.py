"""Main extraction orchestrator.

Coordinates: model inference → dictionary matching → normalization → structured output.
Falls back to Claude teacher if no student model is available.
"""

import logging
import os
import re

from .schema import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    RelationshipContext,
)
from .model.inference import extract_with_student, is_model_available
from .model.ensemble import extract_ensemble
from .dictionaries.lookup import DictionaryLookup
from .normalize import EntityNormalizer

logger = logging.getLogger(__name__)


# Relationship types that are structural (variant is in gene), not functional
_STRUCTURAL_REL_TYPES = {"located_in", "interacts_with", "associated_with"}


def _is_structural_variant_gene(subj_type: str, obj_type: str, rel_type: str) -> bool:
    """Check if this is a structural variant-gene relationship.

    BioRED does not annotate "variant is located in gene" — these are structural
    facts, not functional relationships. Filter them out.
    """
    types = {subj_type, obj_type}
    if "VARIANT" not in types:
        return False
    if "GENE" not in types and "PROTEIN" not in types:
        return False
    return rel_type.lower() in _STRUCTURAL_REL_TYPES


def _is_metabolite_relationship(
    subj: str, obj: str, subj_type: str, obj_type: str,
) -> bool:
    """Check if this is a metabolite/conversion relationship.

    Chemical <-> chemical where one name is clearly derived from the other
    (e.g., "doxorubicin" <-> "doxorubicinol", "metoprolol" <-> "alpha-hydroxymetoprolol").
    """
    chemical_types = {"CHEMICAL", "DRUG"}
    if subj_type not in chemical_types or obj_type not in chemical_types:
        return False

    shorter, longer = (subj, obj) if len(subj) <= len(obj) else (obj, subj)
    # One chemical name is a substring of the other (metabolite naming convention)
    if shorter in longer and len(shorter) >= 5:
        return True

    return False


def _is_hypernym_relationship(subj: str, obj: str, subj_type: str, obj_type: str) -> bool:
    """Check if this is a hypernym/class-membership relationship.

    E.g., "doxorubicin" <-> "anthracyclines" where one is a member of the class.
    These are taxonomic facts, not functional relationships.
    """
    # Both must be the same broad type
    if subj_type != obj_type:
        return False

    # Check if one is a plural/class form containing the other's root
    # "anthracycline" in "anthracyclines", "doxorubicin" is-a "anthracycline"
    # Heuristic: if one entity name appears as a substring of common class suffixes
    shorter, longer = (subj, obj) if len(subj) <= len(obj) else (obj, subj)

    # Plural class names: "anthracyclines" -> "anthracycline"
    longer_singular = longer.rstrip("s")

    # Check if shorter is a known specific instance of the longer class
    # This is conservative: only trigger when shorter contains the class root
    # e.g., "doxorubicin" doesn't contain "anthracycline" — so we need another heuristic

    # Simpler: if one name ends with the other (e.g., "beta-blocker" and "blocker")
    if shorter in longer and len(shorter) >= 4:
        return True

    return False


# Too-generic entities that produce FP cascades
_GENERIC_ENTITIES = {
    # Generic biomolecules
    "dna", "rna", "mrna", "cdna", "protein", "gene", "peptides", "peptide",
    "amino acid", "amino acids", "nucleotide", "nucleotides",
    "nucleoside", "nucleosides", "purine deoxyribonucleosides",
    "deoxyribonucleosides", "ribonucleosides",
    # Generic drug classes / chemical terms
    "beta-blocker", "beta-blockers", "opioid drugs", "opioid",
    "cancer therapies", "chemotherapy",
    "r-isomer", "s-isomer", "isomer", "enantiomer",
    "metabolite", "metabolites", "substrate", "substrates",
    # Generic terms
    "mutation", "mutations", "polymorphism", "polymorphisms",
    "wild-type", "wild type",
    # Generic receptor/enzyme classes
    "opioid receptor", "receptor", "enzyme", "channel",
    # Too-vague disease terms
    "pain perception", "addiction",
    # Study population descriptors, not entities
    "childhood cancer survivors", "childhood cancer survivor",
    "cancer survivors", "cancer survivor",
    # Biological processes, not diseases
    "oxidative stress", "lipid peroxidation", "apoptosis", "necrosis",
    "inflammation", "fibrosis", "cell death", "cell proliferation",
    # Generic receptor/enzyme/channel classes (already above)
    # Too-generic gene/protein terms
    "water channel", "dopamine receptor",
    # Measurement/phenotype terms, not diseases
    "fat mass", "body weight", "body mass",
    # Consumption/exposure terms, not chemicals
    "alcohol consumption", "chronic alcohol consumption",
    "drug consumption", "smoking",
    # Study design / methodology (MeSH terms that aren't biomedical entities)
    "methods", "risk factors", "risk factor",
    "case-control studies", "case-control study",
    "nested case-control study", "cohort study", "cohort studies",
    "clinical trial", "clinical trials",
    "odds ratio", "confidence interval",
    "prevalence", "incidence", "mortality", "morbidity",
    "isoform", "isoforms", "protein isoforms",
    "codon", "codons", "exon", "exons", "intron", "introns",
    "pharmacokinetics", "bioavailability", "half-life",
    "body mass index",
}

# Disease subtype pattern: "LQTS-3", "LQTS-2", "Type 1", etc.
_DISEASE_SUBTYPE_RE = re.compile(
    r'^(.+?)[-\s]+(\d+|[IVX]+|type\s+\d+)$', re.IGNORECASE
)


def _is_too_generic(text: str, entity_type: str) -> bool:
    """Check if an entity is too generic or too descriptive to be useful."""
    t = text.lower().strip()
    if t in _GENERIC_ENTITIES:
        return True

    # Filter long descriptive disease phrases (radiological/clinical findings)
    # Real disease entities are typically 1-4 words. Longer phrases like
    # "hyperintensity in the bilateral pallidi" are descriptive findings.
    if entity_type == "DISEASE" and len(t.split()) >= 5:
        # Allow known long disease names
        if not any(term in t for term in [
            "syndrome", "disease", "disorder", "deficiency",
            "depletion", "failure", "cancer", "carcinoma",
        ]):
            return True

    return False


def _is_disease_subtype(text: str, all_entity_texts: set[str]) -> bool:
    """Check if entity is a disease subtype or qualified form when parent exists.

    E.g., "LQTS-3" when "LQTS" exists, "fetal bradycardia" when "bradycardia" exists.
    """
    t = text.lower().strip()

    # Numeric subtype: "LQTS-3", "Type 2 diabetes"
    m = _DISEASE_SUBTYPE_RE.match(text)
    if m:
        base = m.group(1).strip().lower()
        if base in all_entity_texts:
            return True

    # Qualified form: "fetal X", "childhood X", "neonatal X", "combined X", etc.
    qualifiers = [
        "fetal ", "neonatal ", "childhood ", "infantile ", "juvenile ",
        "congenital ", "familial ", "hereditary ", "acquired ",
        "combined ", "acute ", "chronic ", "severe ", "mild ",
        "2:1 ", "1:1 ",
    ]
    for q in qualifiers:
        if t.startswith(q):
            base = t[len(q):]
            if base in all_entity_texts:
                return True

    return False


# Compound variant entities: gene name + variant notation combined
# e.g., "CBR3 V244M", "NQO1*2", "SCN5A R1232W"
_COMPOUND_VARIANT_RE = re.compile(
    r'^[A-Z][A-Z0-9]{1,10}\s+[A-Z]\d+[A-Z]$'  # "CBR3 V244M"
    r'|^[A-Z][A-Z0-9]{1,10}\*\d+$'              # "NQO1*2"
)

# HBV/HIV domain-prefixed variants: "rtL180M" → "L180M", "sE164D" → "E164D"
# Common prefixes: rt (reverse transcriptase), s (surface), pre (precore)
_DOMAIN_PREFIX_VARIANT_RE = re.compile(
    r'^(rt|pre|s)([A-Z]\d+[A-Z])$', re.IGNORECASE
)


class BioExtractor:
    """Main extraction engine.

    Extracts biomedical entities and relationships from text, normalizes
    mentions to canonical IDs, and returns structured results.
    """

    def __init__(self):
        self._dictionary = DictionaryLookup()
        self._normalizer = EntityNormalizer(self._dictionary)
        self._use_teacher = not is_model_available()

        if self._use_teacher:
            logger.info("No student model found — will use Claude teacher for extraction")
        else:
            logger.info("Student model loaded — using local extraction")

    def extract(self, text: str, use_ensemble: bool = False) -> ExtractionResult:
        """Extract entities and relationships from a single text.

        Args:
            text: Biomedical abstract text.
            use_ensemble: If True, use the 5-model ensemble pipeline.
                Otherwise, use single-model extraction.
        """
        if use_ensemble:
            return self._extract_with_ensemble(text)

        # Single-model path (original)
        # Step 1: Raw extraction
        if self._use_teacher:
            result = self._extract_with_teacher(text)
        else:
            result = extract_with_student(text)
            if result is None:
                result = self._extract_with_teacher(text)

        if result is None:
            return ExtractionResult(text=text)

        # Step 2: Normalize entities to canonical IDs
        if self._dictionary.is_available():
            result = self._normalizer.normalize_result(result)

        # Step 3: Post-extraction improvements
        result = self._filter_bad_entities(result)
        result = self._deduplicate_entities(result)
        result = self._filter_bad_relationships(result)

        return result

    def _extract_with_ensemble(self, text: str) -> ExtractionResult:
        """Run the ensemble extraction pipeline."""
        result = extract_ensemble(
            text,
            dictionary=self._dictionary,
            normalizer=self._normalizer,
            skip_verifier=os.environ.get("BIOEXTRACT_SKIP_VERIFIER") == "1",
        )

        # Apply post-extraction cleanup
        result = self._filter_bad_entities(result)
        result = self._deduplicate_entities(result)
        result = self._filter_bad_relationships(result)

        return result

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract from multiple texts."""
        return [self.extract(text) for text in texts]

    def _extract_with_teacher(self, text: str) -> ExtractionResult | None:
        """Use Claude API as a teacher model for extraction.

        Runs the ensemble pipeline with n_runs=1 (single Sonnet call for
        relationships) as a lightweight fallback when no student model exists.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "No ANTHROPIC_API_KEY set and no student model available. "
                "Cannot perform extraction."
            )
            return None

        try:
            return extract_ensemble(
                text,
                dictionary=self._dictionary,
                normalizer=self._normalizer,
                n_runs=1,
            )
        except Exception as e:
            logger.error("Teacher extraction failed: %s", e)
            return None

    def _filter_bad_entities(self, result: ExtractionResult) -> ExtractionResult:
        """Remove entities that are known to be problematic.

        Filters:
        - Compound variant entities (gene+variant combined, e.g., "CBR3 V244M", "NQO1*2")
        - Too-generic entities that produce false positive relationships
        - Disease subtype notations (e.g., "LQTS-3")
        - Duplicate full-name entities when abbreviation already exists
        """
        if not result.entities:
            return result

        # Collect all entity texts for dedup checking
        entity_texts = {e.text.lower().strip() for e in result.entities}

        bad_keys: set[str] = set()
        rename_map: dict[str, str] = {}  # old_text_lower -> new_text
        good_entities = []
        for e in result.entities:
            text_lower = e.text.lower().strip()
            etype = e.type.upper()

            # Filter compound variant entities
            if etype == "VARIANT" and _COMPOUND_VARIANT_RE.match(e.text):
                bad_keys.add(text_lower)
                continue

            # Strip domain prefixes from variants: "rtL180M" → "L180M"
            if etype == "VARIANT":
                m = _DOMAIN_PREFIX_VARIANT_RE.match(e.text)
                if m:
                    old_text = e.text
                    e.text = m.group(2)
                    text_lower = e.text.lower().strip()
                    # Track rename so we can update relationship texts
                    rename_map[old_text.lower().strip()] = e.text

            # Filter too-generic entities
            if _is_too_generic(e.text, etype):
                bad_keys.add(text_lower)
                continue

            # Filter disease subtype notations like "LQTS-3" when "LQTS" exists
            if etype == "DISEASE" and _is_disease_subtype(e.text, entity_texts):
                bad_keys.add(text_lower)
                continue

            good_entities.append(e)

        if not bad_keys and not rename_map:
            return result

        # Update relationship texts for renamed entities, and remove bad ones
        good_rels = []
        for r in result.relationships:
            subj_key = r.subject.lower().strip()
            obj_key = r.object.lower().strip()
            if subj_key in bad_keys or obj_key in bad_keys:
                continue
            # Apply renames from variant prefix stripping
            new_subj = rename_map.get(subj_key, r.subject)
            new_obj = rename_map.get(obj_key, r.object)
            if new_subj != r.subject or new_obj != r.object:
                r = ExtractedRelationship(
                    subject=new_subj,
                    predicate=r.predicate,
                    object=new_obj,
                    type=r.type,
                    direction=r.direction,
                    negated=r.negated,
                    context=r.context,
                    confidence=r.confidence,
                )
            good_rels.append(r)

        return ExtractionResult(
            text=result.text,
            entities=good_entities,
            relationships=good_rels,
            extraction_method=result.extraction_method,
        )

    def _deduplicate_entities(self, result: ExtractionResult) -> ExtractionResult:
        """Merge entities that resolve to the same canonical ID.

        When multiple text mentions (e.g., "LQTS" and "long QT syndrome") normalize
        to the same canonical_id, keep all as entities but build a mapping so
        relationships referencing any variant use the canonical name consistently.
        """
        if not result.entities:
            return result

        # Build canonical_id -> list of entity texts
        canonical_groups: dict[str, list[ExtractedEntity]] = {}
        for e in result.entities:
            if e.canonical_id:
                canonical_groups.setdefault(e.canonical_id, []).append(e)

        # Build text -> canonical representative text mapping
        # The representative is the entity with the longest text (most descriptive)
        text_alias: dict[str, str] = {}  # lowered text -> canonical text
        for cid, group in canonical_groups.items():
            if len(group) <= 1:
                continue
            # Pick the longest name as canonical representative
            representative = max(group, key=lambda e: len(e.text))
            for e in group:
                if e.text != representative.text:
                    text_alias[e.text.lower()] = representative.text

        if not text_alias:
            return result

        # Update relationship subject/object to use canonical text
        updated_rels = []
        seen_rels = set()
        for r in result.relationships:
            subj = text_alias.get(r.subject.lower(), r.subject)
            obj = text_alias.get(r.object.lower(), r.object)
            # Skip self-references created by aliasing
            if subj.lower() == obj.lower():
                continue
            # Deduplicate relationships that become identical after aliasing
            rel_key = (subj.lower(), obj.lower(), r.type)
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                updated_rels.append(ExtractedRelationship(
                    subject=subj,
                    predicate=r.predicate,
                    object=obj,
                    type=r.type,
                    direction=r.direction,
                    negated=r.negated,
                    context=r.context,
                    confidence=r.confidence,
                ))

        return ExtractionResult(
            text=result.text,
            entities=result.entities,  # Keep all entity mentions
            relationships=updated_rels,
            extraction_method=result.extraction_method,
        )

    def _filter_bad_relationships(self, result: ExtractionResult) -> ExtractionResult:
        """Filter out obviously bad relationships.

        Removes:
        - Orphaned relationships (subject/object not in entity list)
        - Self-referencing relationships (subject == object after normalization)
        - Relationships where both entities resolve to the same canonical ID
        - Structural variant-gene relationships (variant "located_in" gene, etc.)
        - Metabolite/conversion relationships (chemical <-> chemical substring)
        """
        if not result.relationships:
            return result

        # Build entity lookups using normalized keys
        entity_canonical: dict[str, str] = {}
        entity_type: dict[str, str] = {}
        entity_keys: set[str] = set()  # all valid entity text keys
        for e in result.entities:
            key = " ".join(e.text.lower().split())
            entity_keys.add(key)
            if e.canonical_id:
                entity_canonical[key] = e.canonical_id
            entity_type[key] = e.type.upper()

        # Also build canonical_id lookup by canonical_id -> set of text keys
        cid_to_texts: dict[str, set[str]] = {}
        for e in result.entities:
            if e.canonical_id:
                nkey = " ".join(e.text.lower().split())
                cid_to_texts.setdefault(e.canonical_id, set()).add(nkey)

        def _norm(s: str) -> str:
            """Normalize text for comparison: lowercase, collapse whitespace."""
            return " ".join(s.lower().split())

        def _find_entity_key(text: str) -> str | None:
            """Find matching entity key with fuzzy fallback."""
            if text in entity_keys:
                return text
            # Try substring match — require high overlap to avoid false matches
            best_match = None
            best_ratio = 0.0
            for ek in entity_keys:
                if text in ek or ek in text:
                    shorter = min(len(text), len(ek))
                    longer = max(len(text), len(ek))
                    ratio = shorter / longer if longer > 0 else 0
                    # Require at least 60% overlap to count as a match
                    if ratio > 0.6 and ratio > best_ratio:
                        best_match = ek
                        best_ratio = ratio
            return best_match

        filtered = []
        for r in result.relationships:
            subj = _norm(r.subject)
            obj = _norm(r.object)

            # Skip orphaned relationships (subject/object not in entity list)
            subj_key = _find_entity_key(subj)
            obj_key = _find_entity_key(obj)
            if subj_key is None or obj_key is None:
                continue

            # Skip self-referencing (text match)
            if subj == obj:
                continue

            # Skip self-referencing (both sides resolve to same entity)
            if subj_key == obj_key:
                continue

            # Skip self-referencing (substring match for near-duplicates)
            if subj in obj or obj in subj:
                shorter, longer = (subj, obj) if len(subj) <= len(obj) else (obj, subj)
                if len(shorter) / len(longer) > 0.5:
                    continue

            # Skip if both resolve to same canonical ID
            subj_cid = entity_canonical.get(subj) or entity_canonical.get(subj_key)
            obj_cid = entity_canonical.get(obj) or entity_canonical.get(obj_key)
            if subj_cid and obj_cid and subj_cid == obj_cid:
                continue

            # Skip if subject or object text maps to the same canonical_id
            # (handles cases where relationship text doesn't exactly match entity text)
            if not subj_cid:
                for cid, texts in cid_to_texts.items():
                    if subj in texts or subj_key in texts:
                        subj_cid = cid
                        break
            if not obj_cid:
                for cid, texts in cid_to_texts.items():
                    if obj in texts or obj_key in texts:
                        obj_cid = cid
                        break
            if subj_cid and obj_cid and subj_cid == obj_cid:
                continue

            # Skip structural variant-gene relationships
            subj_t = entity_type.get(subj_key, entity_type.get(subj, ""))
            obj_t = entity_type.get(obj_key, entity_type.get(obj, ""))
            if _is_structural_variant_gene(subj_t, obj_t, r.type):
                continue

            # Skip metabolite/conversion relationships (chemical <-> chemical substring)
            if _is_metabolite_relationship(subj, obj, subj_t, obj_t):
                continue

            # Skip hypernym/class-membership relationships
            if _is_hypernym_relationship(subj, obj, subj_t, obj_t):
                continue

            filtered.append(r)

        if len(filtered) < len(result.relationships):
            return ExtractionResult(
                text=result.text,
                entities=result.entities,
                relationships=filtered,
                extraction_method=result.extraction_method,
            )
        return result

    @property
    def status(self) -> dict:
        """Return service status."""
        return {
            "model": "student" if not self._use_teacher else "claude_teacher",
            "model_available": is_model_available(),
            "dictionary_available": self._dictionary.is_available(),
            "dictionary_stats": self._dictionary.stats() if self._dictionary.is_available() else None,
        }
