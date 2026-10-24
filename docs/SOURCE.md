# Public source snapshot

This release exposes the accepted research mechanism and its frozen helper dependencies. [source-manifest.json](source-manifest.json) records 41 exact copied Python files, their SHA256 values and accepted boundaries. Original research snapshots remain unchanged. New documentation, packaging checks and the tensor exporter are release work, not historical experimental source.

| Location | Purpose |
|---|---|
| `swarm/population_objective.py` | Causal K2/K4 mixture objective and recomputed gradients |
| `swarm/population_runtime.py` | Serial member inference, independent caches and bounded generation |
| `swarm/population_assembly.py` | Exact formed-artifact validation and four-member assembly |
| `swarm/formation_worker.py` | Bounded supervised fork formation, update/replay accounting and restoration |
| `swarm/collective_*.py` | Collective controller, numerical admission, recovery and resource checks |
| `swarm/collective_trust.py` | Accepted explicit post-update trust extension |
| `research_runtime/` | Unmodified historical helpers, relocated into a visible public directory |
| `tools/export_release_weights.py` | New tensor-only serialization and exact roundtrip check |
| `docs/weights/` | Source-checkpoint lineage and downloadable asset hashes |

The trust extension must be instantiated explicitly by a future training package. Existing workers do not acquire it merely because its source is present. Accepted local software and synthetic verification do not establish actual pretrained K4 speed, memory, recovery cost or learning benefit.

The repository also retains its original small-model code and tests from public commit `e913c988c2b9d7c0c793046d3588dc2c8af7c0a6`. That model is separate from the newer pretrained population runtime. Private unfinished learner edits are preserved in the research workspace and are not substituted into this public implementation.

## Checking the checkout

Python 3.11 or later is recommended. The standard-library packaging check needs no model dependencies:

```sh
python tools/check_release.py
```

For module imports, use an environment with the dependencies in `requirements-research.txt`. The recorded GPU environment and serialization environment are different; see the model card. Historical helpers retain bare imports, so include their directory explicitly:

```sh
PYTHONPATH=research_runtime:. python tools/check_release.py --imports
PYTHONPATH=research_runtime:. python -m unittest discover -s tests -p test_public_contracts.py
```

The second command only tests probability arithmetic and validation using fixed small tensors. It performs no model construction, language-model inference, optimizer step, dataset access or provider call. This is a packaging check, not a repeat of the historical experimental audits.

The release preparation passed all 41 source identity/import checks and both fixed-tensor unit tests. [Recorded environment and results](release-checks.json).

## Limits of the public package

Provider entry scripts, account-specific launch glue, raw datasets, private prompts, benchmark gold/tests and operational receipts are excluded. Some preserved admission functions expect the original GPU layout and files under `/kaggle/input`; there is no automatic launch command in this release. The exact frozen NF4 base and private evaluation fixtures are not included. Import success does not prove portable inference, and the new safetensors assets still require loader integration.

The accepted K4 admission source and accepted trust extension were selected deliberately; the separate live transport candidate was not included. This source release does not claim that a K4 training run has completed.
