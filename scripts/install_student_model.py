"""Install the student model downloaded from Colab.

Extracts student_model.zip and places the NER and RE models into
bioextract/model/trained/ so BioExtractor picks them up automatically.

Usage:
    python scripts/install_student_model.py ~/Downloads/student_model.zip
"""

import shutil
import sys
import zipfile
from pathlib import Path

TRAINED_DIR = Path(__file__).parent.parent / "bioextract" / "model" / "trained"


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/install_student_model.py <path/to/student_model.zip>")
        sys.exit(1)

    zip_path = Path(sys.argv[1]).expanduser()
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found")
        sys.exit(1)

    print(f"Installing student model from {zip_path}...")

    # Extract to a temp directory inside the zip's folder
    tmp_dir = zip_path.parent / "_student_model_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    # Locate ner/ and re/ inside the extracted tree
    ner_src = tmp_dir / "ner"
    re_src = tmp_dir / "re"

    if not ner_src.exists() or not re_src.exists():
        print(f"ERROR: expected ner/ and re/ inside zip, found: {list(tmp_dir.iterdir())}")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    # Install
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    ner_dest = TRAINED_DIR / "ner"
    re_dest = TRAINED_DIR / "re"

    if ner_dest.exists():
        shutil.rmtree(ner_dest)
    if re_dest.exists():
        shutil.rmtree(re_dest)

    shutil.copytree(ner_src, ner_dest)
    shutil.copytree(re_src, re_dest)
    shutil.rmtree(tmp_dir)

    print(f"NER model installed to {ner_dest}")
    print(f"RE  model installed to {re_dest}")
    print("\nVerifying...")

    from bioextract.model.inference import is_model_available
    if is_model_available():
        print("Student model is ready. BioExtractor will use it automatically.")
    else:
        print("WARNING: model files found but is_model_available() returned False.")
        print("Check that config.json and re_heads.pt are present.")


if __name__ == "__main__":
    main()
