"""Convert BioRED BioC JSON to JSONL training/evaluation format."""

import json
import sys
from pathlib import Path

# Map BioRED entity types to our schema types
ENTITY_TYPE_MAP = {
    "GeneOrGeneProduct": "GENE",
    "DiseaseOrPhenotypicFeature": "DISEASE",
    "ChemicalEntity": "CHEMICAL",
    "OrganismTaxon": "ORGANISM",
    "SequenceVariant": "VARIANT",
    "CellLine": "CELL_TYPE",
}

# Map BioRED relation types to our schema types
REL_TYPE_MAP = {
    "Association": "associated_with",
    "Bind": "binds",
    "Positive_Correlation": "upregulates",
    "Negative_Correlation": "downregulates",
    "Drug_Interaction": "interacts_with",
    "Cotreatment": "interacts_with",
    "Conversion": "regulates",
    "Comparison": "associated_with",
}

# Map BioRED relation types to direction
REL_DIRECTION_MAP = {
    "Positive_Correlation": "positive",
    "Negative_Correlation": "negative",
    "Bind": "neutral",
    "Association": "neutral",
    "Drug_Interaction": "neutral",
    "Cotreatment": "neutral",
    "Conversion": "positive",
    "Comparison": "neutral",
}


def convert_document(doc: dict) -> dict | None:
    """Convert a single BioRED document to our JSONL format."""
    # Combine title + abstract text
    passages = doc.get("passages", [])
    text_parts = []
    all_annotations = []

    for passage in passages:
        text_parts.append(passage.get("text", ""))
        all_annotations.extend(passage.get("annotations", []))

    full_text = " ".join(text_parts)
    if not full_text.strip():
        return None

    # Build annotation ID -> info map for relationship resolution
    id_to_info = {}
    entities = []
    seen_entities = set()

    for ann in all_annotations:
        ann_id = ann.get("infons", {}).get("identifier", "")
        ann_type = ann.get("infons", {}).get("type", "")
        ann_text = ann.get("text", "")
        locations = ann.get("locations", [])

        mapped_type = ENTITY_TYPE_MAP.get(ann_type, ann_type)

        # Store for relationship resolution
        if ann_id:
            id_to_info[ann_id] = {
                "text": ann_text,
                "type": mapped_type,
                "identifier": ann_id,
            }

        # Deduplicate entities by text+type
        key = (ann_text.lower(), mapped_type)
        if key not in seen_entities:
            seen_entities.add(key)
            entity = {
                "text": ann_text,
                "type": mapped_type,
            }
            if locations:
                entity["start"] = locations[0].get("offset", 0)
                entity["end"] = locations[0].get("offset", 0) + locations[0].get("length", 0)
            if ann_id:
                entity["identifier"] = ann_id
            entities.append(entity)

    # Convert relationships
    relationships = []
    for rel in doc.get("relations", []):
        infons = rel.get("infons", {})
        entity1_id = infons.get("entity1", "")
        entity2_id = infons.get("entity2", "")
        rel_type = infons.get("type", "")

        entity1 = id_to_info.get(entity1_id)
        entity2 = id_to_info.get(entity2_id)

        if not entity1 or not entity2:
            continue

        mapped_type = REL_TYPE_MAP.get(rel_type, rel_type.lower())
        direction = REL_DIRECTION_MAP.get(rel_type, "neutral")

        relationships.append({
            "subject": entity1["text"],
            "object": entity2["text"],
            "type": mapped_type,
            "direction": direction,
            "novel": infons.get("novel", ""),
        })

    return {
        "text": full_text,
        "pmid": doc.get("id", ""),
        "entities": entities,
        "relationships": relationships,
        "source": "biored",
    }


def convert_file(input_path: str, output_path: str) -> int:
    """Convert a BioRED BioC JSON file to JSONL."""
    with open(input_path) as f:
        data = json.load(f)

    count = 0
    with open(output_path, "w") as out:
        for doc in data.get("documents", []):
            result = convert_document(doc)
            if result:
                out.write(json.dumps(result) + "\n")
                count += 1

    return count


def main():
    biored_dir = Path("bioextract/BioRED")

    if not biored_dir.exists():
        print(f"BioRED directory not found at {biored_dir}")
        sys.exit(1)

    output_dir = Path("bioextract/data")
    output_dir.mkdir(exist_ok=True)

    for split in ["Train", "Dev", "Test"]:
        input_file = biored_dir / f"{split}.BioC.JSON"
        if not input_file.exists():
            print(f"Skipping {split}: {input_file} not found")
            continue

        output_file = output_dir / f"biored_{split.lower()}.jsonl"
        count = convert_file(str(input_file), str(output_file))
        print(f"{split}: {count} documents -> {output_file}")

    # Print sample from test set
    test_out = output_dir / "biored_test.jsonl"
    if test_out.exists():
        with open(test_out) as f:
            sample = json.loads(f.readline())
        print(f"\nSample (PMID {sample['pmid']}):")
        print(f"  Text: {sample['text'][:100]}...")
        print(f"  Entities: {len(sample['entities'])}")
        print(f"  Relationships: {len(sample['relationships'])}")
        if sample["entities"]:
            print(f"  First entity: {sample['entities'][0]}")
        if sample["relationships"]:
            print(f"  First rel: {sample['relationships'][0]}")


if __name__ == "__main__":
    main()
