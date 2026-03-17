"""Entity and relationship type definitions for biomedical extraction."""

from dataclasses import dataclass, field
from enum import Enum


class EntityType(str, Enum):
    GENE = "GENE"
    PROTEIN = "PROTEIN"
    DISEASE = "DISEASE"
    DRUG = "DRUG"
    CHEMICAL = "CHEMICAL"
    PATHWAY = "PATHWAY"
    CELL_TYPE = "CELL_TYPE"
    TISSUE = "TISSUE"
    ORGANISM = "ORGANISM"
    BIOLOGICAL_PROCESS = "BIOLOGICAL_PROCESS"
    MOLECULAR_FUNCTION = "MOLECULAR_FUNCTION"
    PHENOTYPE = "PHENOTYPE"
    VARIANT = "VARIANT"
    ANATOMICAL_STRUCTURE = "ANATOMICAL_STRUCTURE"


class RelationshipType(str, Enum):
    ACTIVATES = "activates"
    INHIBITS = "inhibits"
    UPREGULATES = "upregulates"
    DOWNREGULATES = "downregulates"
    ASSOCIATED_WITH = "associated_with"
    CAUSES = "causes"
    TREATS = "treats"
    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"
    BINDS = "binds"
    PHOSPHORYLATES = "phosphorylates"
    EXPRESSED_IN = "expressed_in"
    LOCATED_IN = "located_in"
    REGULATES = "regulates"
    INTERACTS_WITH = "interacts_with"


@dataclass
class ExtractedEntity:
    """A raw entity mention extracted from text."""
    text: str
    type: str  # EntityType value
    start: int  # character offset
    end: int
    canonical_id: str | None = None
    canonical_name: str | None = None
    confidence: float = 1.0


@dataclass
class RelationshipContext:
    """Contextual metadata for an extracted relationship."""
    organism: str | None = None
    cell_type: str | None = None
    experiment_type: str | None = None  # in_vitro, in_vivo, clinical, computational


@dataclass
class ExtractedRelationship:
    """A relationship triple extracted from text."""
    subject: str  # entity text
    predicate: str  # raw verb from text
    object: str  # entity text
    type: str  # RelationshipType value
    direction: str  # positive, negative, neutral
    negated: bool = False
    context: RelationshipContext = field(default_factory=RelationshipContext)
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    """Complete extraction output for a single text."""
    text: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    extraction_method: str = "bioextract_v1"
