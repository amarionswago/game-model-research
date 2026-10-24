# Upstream attribution and license scope

The root MIT license covers original HIVE contributions, including its released adapter contributions. It does not relicense upstream models, library code, datasets, trademarks or other third-party material. Dependencies and pretrained base weights are obtained separately.

- **GLM-4.7-Flash:** Z.ai / GLM Team. The [pinned upstream model card](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/7dd20894a642a0aa287e9827cb1a1f7f91386b67/README.md) declares `license: mit`. The [pinned file tree](https://huggingface.co/zai-org/GLM-4.7-Flash/tree/7dd20894a642a0aa287e9827cb1a1f7f91386b67) has no separate LICENSE file. HIVE distributes adapter deltas and a gate, not the base model. The separate [upstream GLM code repository](https://github.com/zai-org/GLM-4.5/blob/main/LICENSE) uses Apache-2.0; its notice is retained in [licenses/GLM-code-Apache-2.0.txt](licenses/GLM-code-Apache-2.0.txt). This code license is not substituted for the model-card declaration.
- **Muon:** Keller Jordan and contributors. `research_runtime/muon_optimizer.py` implements the documented Nesterov / quintic Newton–Schulz method and the PyTorch `match_rms_adamw` scaling convention, with FP32 arithmetic for T4. [Upstream source](https://github.com/KellerJordan/Muon) and [MIT notice](licenses/Muon-MIT.txt). The numerical implementation is a project variant, not a claim of bitwise equivalence to every upstream implementation.
- **Transformers and PEFT:** Hugging Face and contributors, Apache-2.0. HIVE imports their model and adapter implementations and supplies project numerical/runtime adaptations. See [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE) and [PEFT](https://github.com/huggingface/peft/blob/main/LICENSE).
- **PyTorch:** PyTorch contributors, BSD-style license. See [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE).
- **bitsandbytes:** bitsandbytes contributors, MIT. See [bitsandbytes license](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/LICENSE).
- **Safetensors:** Hugging Face and contributors, Apache-2.0. See [Safetensors license](https://github.com/huggingface/safetensors/blob/main/LICENSE).

Training corpora, task statements, teacher solutions and benchmark tests are not redistributed. References to MATH-500 and HumanEval identify evaluation sources, not a license grant. Historical exposure and dataset provenance limitations are described in [MODEL_CARD.md](MODEL_CARD.md) and [docs/RESEARCH.md](docs/RESEARCH.md).

Source and license links were checked on October 24, 2026. No third-party endorsement or academic peer review is implied.
