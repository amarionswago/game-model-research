# HIVE

**Can language models learn to be more useful together?**

HIVE investigates collective prediction and learning by whole language-model members. Each member has its own trainable adapter over a shared frozen pretrained backbone. Members read the same prefix, contribute next-token probabilities, and produce one answer stream. Training can optimize that collective prediction directly.

The research asks when this cooperation adds capability, what each member contributes, and whether any gains justify the full cost of training and inference. HIVE is an independent, AI-assisted research project led by [Amarion Fred](https://github.com/amarionfred).

**October 24, 2026:** K2 training and evaluation are complete at the scopes below. Four additional supervised adapter endpoints have been formed and evaluated. K4 numerical admission and controlled collective learning are ongoing; a cooperation advantage has not yet been demonstrated.

[Research results](docs/RESEARCH.md) · [Adapter weights and model card](MODEL_CARD.md) · [Weight releases](https://github.com/amarionfred/swarm-model-research/releases)

## What has been built

- A shared-base runtime with separate LoRA members, collective decoding, learned gating and direct collective-loss training.
- Real GPU training: 202 retained individual updates, 32 subsequent joint updates, an eight-update Muon pilot, and a later comparison of joint, independent, uniform-mixture and single-member training.
- Saved checkpoints with numerical checks, immutable-base checks, recovery and replay accounting.
- Maths and coding evaluation with per-task outcomes, explicit output limits, reused-answer tracking and failed-run costs.
- Four separately trained forks of the same parent member: arithmetic, sequence/mapping, symbolic/discrete and tree operations. These are curriculum labels, not established specialties.

## What the experiments show

| Experiment | Observed result | Scope |
|---|---|---|
| Benchmark-derived sample | Maths **6/12**, coding **10/12** | MATH-500 and HumanEval subsets; not full-suite scores |
| Member contribution | Best solo **10/12**, joint **8/12**, independent **7/12** | Same twelve exposed tasks; joint adds no success beyond the solo union |
| Controlled learning | Joint and independent both **2/6** | Seven matched rounds; evaluation incomplete beyond six common questions |
| Correction pilot | Each of four routes **2/6** | No observed wrong-to-right repairs |
| Further formation | All measured parent successes retained | Small completion gains in arithmetic/symbolic; no demonstrated complementary capability |

The project has produced working systems and interpretable evidence, including negative results. Lower training loss is evidence of fitting; it is not substituted for stronger reasoning or a cooperation benefit. [Detailed results, costs and next comparisons](docs/RESEARCH.md).

## Weights and source

The release contains trained **adapter artifacts**, not a new base model trained from scratch. The backbone is `zai-org/GLM-4.7-Flash` at revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`: roughly 30B total base parameters with roughly 3B active per member. The evaluated K2 system has about **29.985B unique parameters**, including its shared base, two adapters and gate. Running two members still adds compute and cache cost.

Read [MODEL_CARD.md](MODEL_CARD.md) for exact formats, dependencies, checkpoint identities and limitations. The custom research workers include provider-specific paths and admission requirements. Downloadable weights do not by themselves establish portable deployment or minimum hardware requirements.

The repository retains its early small-model engineering code alongside the later pretrained-adapter work. The [source guide](docs/SOURCE.md) explains the 41 exact research modules, [source manifest](docs/source-manifest.json), and checks that run without a model. Private operational notes, raw training corpora and benchmark gold/tests are not part of this release.

Original HIVE contributions are [MIT licensed](LICENSE); [third-party attribution](THIRD_PARTY.md) preserves upstream model and library provenance.

## Authorship

Park supplied the research direction, materials, decisions and experiment authorization, and challenged the interpretation of results. AI coding agents performed substantial implementation, experiment operation, grading and documentation. Separate agent reviews are recorded as agent checks, not external academic peer review.

Upstream models, libraries and research retain their attribution. HIVE's ambition is to test a useful collective-learning mechanism; novelty and larger-model parity remain open questions.
