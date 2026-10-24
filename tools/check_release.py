"""Verify the public source snapshot; optional imports never load a model."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/source-manifest.json").read_text())
    modules = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source identity differs: {entry['path']}")
        ast.parse(raw, filename=entry["path"])
        modules.append(entry["path"][:-3].replace("/", ".")
                       if entry["path"].startswith("swarm/") else path.stem)
    early = manifest["preserved_early_model"]
    if hashlib.sha256((root / early["path"]).read_bytes()).hexdigest() != early["sha256"]:
        raise ValueError("Preserved early model identity differs")
    receipts = sorted((root / "docs/weights").glob("*.json"))
    if len(receipts) != 6:
        raise ValueError("Expected exactly six weight receipts")
    for path in receipts:
        record = json.loads(path.read_text())
        if (record["format"] != "hive-research-tensors-v1"
                or not record["exact_tensor_roundtrip"]
                or record["optimizer_and_rng_included"]):
            raise ValueError(f"Invalid public tensor receipt: {path.name}")
    if args.imports:
        for module in modules:
            importlib.import_module(module)
    print(json.dumps({"source_files_verified": len(modules), "weight_receipts": len(receipts),
                      "imports_checked": args.imports, "model_or_optimizer_calls": 0}))


if __name__ == "__main__":
    main()
