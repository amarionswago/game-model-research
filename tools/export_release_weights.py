"""Export verified HIVE checkpoint tensors, excluding optimizer/RNG/private metadata.

This reads a trusted local checkpoint after its expected SHA256 is verified.
It does not load a base model, run inference, or perform an optimizer update.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(source, expected_sha256, destination):
    source, destination = Path(source), Path(destination)
    receipt = destination.with_suffix(".json")
    if destination.exists() or receipt.exists():
        raise ValueError("Refusing to overwrite an existing export")
    if sha256(source) != expected_sha256:
        raise ValueError("Source checkpoint SHA256 mismatch")
    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    formed = state.get("schema") == "hive-formed-adapter-v1"
    if not formed and state.get("format") != 1:
        raise ValueError("Unsupported checkpoint format")
    tensors = state["adapters"]
    expected_count, expected_parameters = (470, 21031936) if formed else (941, 42067969)
    if len(tensors) != expected_count:
        raise ValueError("Wrong tensor inventory")
    member_counts = {"member0": 0, "member1": 0}
    for name, tensor in tensors.items():
        if name == "mix_gate.weight" and not formed:
            if tensor.numel() != 4097:
                raise ValueError("Wrong K2 gate size")
        else:
            owners = [member for member in member_counts if f".{member}." in name]
            if len(owners) != 1 or ".lora_" not in name or tensor.ndim != 2:
                raise ValueError("Unexpected tensor name or shape")
            member_counts[owners[0]] += 1
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Invalid tensor dtype/value")
    if member_counts != {"member0": 470, "member1": 0 if formed else 470}:
        raise ValueError("Wrong member ownership")
    if sum(t.numel() for t in tensors.values()) != expected_parameters:
        raise ValueError("Wrong parameter count")
    if formed and (state["provenance"].get("retained_updates") != 2 or
                   state["provenance"].get("recovery_verified") is not True):
        raise ValueError("Not an accepted two-update formation endpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.contiguous() for name, tensor in tensors.items()},
              str(destination), metadata={"source_checkpoint_sha256": expected_sha256,
                                         "format": "hive-research-tensors-v1"})
    restored = load_file(str(destination), device="cpu")
    if set(restored) != set(tensors) or not all(
            torch.equal(restored[name], tensor) for name, tensor in tensors.items()):
        raise ValueError("Export failed exact tensor roundtrip verification")
    record = {"file": destination.name, "bytes": destination.stat().st_size,
              "sha256": sha256(destination), "source_checkpoint_sha256": expected_sha256,
              "format": "hive-research-tensors-v1", "tensor_count": expected_count,
              "parameters": expected_parameters, "dtype": "float32",
              "member_tensors": member_counts, "contains_k2_gate": not formed,
              "exact_tensor_roundtrip": True, "optimizer_and_rng_included": False,
              "torch_export_version": torch.__version__}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(export(args.source, args.sha256, args.destination), indent=2))
