"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Entity:
    id: str  # type-prefixed, e.g. "gene:6622"
    type: str  # gene, disease, drug, variant, ...
    name: str
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    external_ids: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class Relationship:
    source_id: str
    target_id: str
    type: str  # associated_with, treats, targets, ...
    source_db: str  # provenance
    confidence: float | None = None
    evidence: dict = field(default_factory=dict)


class Source(ABC):
    """Base class for all data sources."""

    name: str = "unnamed"

    @abstractmethod
    def fetch(self) -> None:
        """Download raw data from the source."""

    @abstractmethod
    def parse(self) -> tuple[list[Entity], list[Relationship]]:
        """Parse raw data into entities and relationships."""
