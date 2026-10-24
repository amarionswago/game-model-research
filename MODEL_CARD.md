# HIVE adapter research release

**Release:** [hive-research-2026-10-24](https://github.com/amarionfred/swarm-model-research/releases/tag/hive-research-2026-10-24). Six tensor-only research artifacts: two K2 adapter populations and four subsequent single-member formation endpoints. They require a separately obtained base model and the matching custom runtime. There is no trained K4 checkpoint in this release.

## Downloads and identities

Download the `.safetensors` files and matching JSON receipts from the release. Each receipt records the new file SHA256, original checkpoint SHA256, exact tensor count and roundtrip result. These are FP32 adapter tensors; optimizer, scheduler, RNG, cursor and private provenance payloads were excluded.

| Asset | Tensor count | Parameters | Size in bytes | Receipt |
|---|---:|---:|---:|---|
| `hive-k2-joint.safetensors` | 941 | 42,067,969 | 168,410,164 | [JSON](docs/weights/hive-k2-joint.json) |
| `hive-k2-independent.safetensors` | 941 | 42,067,969 | 168,410,164 | [JSON](docs/weights/hive-k2-independent.json) |
| `hive-stage2-arithmetic.safetensors` | 470 | 21,031,936 | 84,196,488 | [JSON](docs/weights/hive-stage2-arithmetic.json) |
| `hive-stage2-sequence.safetensors` | 470 | 21,031,936 | 84,196,488 | [JSON](docs/weights/hive-stage2-sequence.json) |
| `hive-stage2-symbolic.safetensors` | 470 | 21,031,936 | 84,196,488 | [JSON](docs/weights/hive-stage2-symbolic.json) |
| `hive-stage2-tree.safetensors` | 470 | 21,031,936 | 84,196,488 | [JSON](docs/weights/hive-stage2-tree.json) |

Original accepted checkpoint identities, before conversion:

```text
joint       3fdd016f85e3840e553acd88e80fc0cab798fbeee40472ad805a317576823fe0
independent 8322a030c1dc43a7a84cb5475426f6e1a2ee854a527bc70236133de83b7ff197
arithmetic  7008724b1a21804e21fe58ed0641799d8bf08ae534b6a01be38e1e4e0ea766cb
sequence    ce2a76fbb8c694856366d7e957c26502a9344c31fd2bd7bd6bb4810ba8cbaf23
symbolic    5a8b785d35234fae52e6860c2d8f1e468e5a19da3db2561b35e73967d11d269a
tree        e3d254a7fdec071c25c3ec7c5db785a565f72959cfc528daa6fc4f1e793bf0fd
```

The exporter verified each original file hash, allowed schema, names, matrix counts, FP32 dtype and finite values; then compared every saved tensor exactly after reopening the safetensors file. The CPU serialization environment was PyTorch 2.14.1+cpu and safetensors 0.8.0. Conversion did not run a language model, train it, or re-evaluate answers. [Exporter source](tools/export_release_weights.py).

## Architecture and lineage

The base is [Z.ai's GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/7dd20894a642a0aa287e9827cb1a1f7f91386b67/README.md), pinned to revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`. Its upstream description is 30B total / 3B active parameters. HIVE shares one frozen base between members and maintains separate adapters and attention caches. This saves base storage; member execution still costs compute.

Each member has rank-16 LoRA, alpha 32, dropout 0 on `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj` and `o_proj` across 47 layers. This produces 470 A/B matrices and 21,031,936 trainable parameters per member. Canonical keys follow:

```text
base_model.model.model.layers.<layer>.self_attn.<projection>.lora_A.member0.weight
base_model.model.model.layers.<layer>.self_attn.<projection>.lora_B.member0.weight
```

Both K2 files contain `.member0.` and `.member1.` tensors plus `mix_gate.weight` (4,097 parameters). The joint and independent controls each retained seven rounds. In the joint arm the collective loss differentiates through member predictions; in the independent control adapter training uses separate member losses. The research runtime combines next-token probabilities, not adapter weights.

All four Stage2 files started independently from the original joint checkpoint's Member0. Each successful screen retained two supervised updates and discarded one exact replay. Their keys are canonical `.member0.` keys. Four-member assembly remaps these to four destination slots and creates a new uniform gate. The files contain no trained K4 gate. Curriculum names describe the intended training focus, not demonstrated expertise.

## Runtime and reproducibility boundary

The accepted GPU runs used a frozen NF4 base export with packed bytes and quantization state retained exactly. This export is not included. Its manifest SHA256 is `f7c7aea54f4a02b4a2300f0eae5e34ae25db9257b2389ef4ef72ddf0529420d6`. Fresh quantization of the same upstream revision has not been shown to reproduce those exact tensors or numerical behavior.

Tokenizer SHA256: `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`. Chat-template SHA256: `d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375`. Thinking was disabled; primary EOS was 154820, with stop IDs 154820/154827/154829.

Captured GPU dependencies were PyTorch 2.11.0+cu128, transformers 5.18.0, PEFT 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, tokenizers 0.23.2 and huggingface_hub 1.31.0. The source includes custom sparse-expert, NF4 and fixed-shape arithmetic repairs and exact-state admission checks. Generic PEFT, vLLM or other inference compatibility has not been verified. Two T4 devices were used for recorded runs; this is not a measured minimum deployment requirement.

The release is suitable for inspecting the mechanism and trained tensors, and for further runtime integration. It is not a self-contained reproduction bundle: the exact NF4 base, private data panels and provider-specific launch package are absent. The preserved research loaders consume the original checkpoint format; they do not automatically consume these new safetensors files. A loader integration must verify names, shapes, base identity and numerical behavior before making equivalent-result claims. [Source layout and checks](docs/SOURCE.md).

You can inspect a downloaded tensor file without loading the base model:

```python
from safetensors import safe_open

with safe_open("hive-k2-joint.safetensors", framework="pt", device="cpu") as weights:
    print(weights.metadata())
    print(len(weights.keys()))  # 941
    print(weights.get_slice("mix_gate.weight").get_shape())
```

Verify the downloaded file's SHA256 against its receipt before use.

## Results and intended use

Use these artifacts for research on collective objectives, adapter contribution, numerical fidelity and cost. The [research record](docs/RESEARCH.md) reports completed scopes and failures. On twelve exposed development tasks, the best solo member solved ten, joint K2 eight and independent K2 seven. Joint added no success beyond the solo union. The four later formation screens retained measured parent successes, with limited completion gains and no comparable graded wrong-to-right repair. Complementarity and a K4 cooperation advantage remain unestablished.

Training used selected mathematical and coding material with qualified source solutions, declared family/exposure exclusions and target-only supervision. Raw corpora, teacher solutions, task panels and benchmark gold/tests are not redistributed. Historical global exposure and pretraining contamination are unknown. The small, exposed evaluations are not full-suite benchmark scores, and do not establish generalization, safety, correctness on arbitrary tasks or parity with larger models.

## Authorship and licensing

Amarion Fred (Park) directed this independent, AI-assisted project. AI agents contributed substantial implementation, operation, review and documentation; agent review is not external academic peer review.

Original HIVE code, documentation and released adapter contributions are offered under [MIT](LICENSE). The upstream model card declares the base model MIT; its pinned repository does not contain a separate model-specific license file. The separate upstream GLM GitHub code repository is Apache-2.0. Neither the base model nor training datasets are redistributed here, and the HIVE license does not replace third-party rights or notices. See [THIRD_PARTY.md](THIRD_PARTY.md).
