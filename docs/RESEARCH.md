# HIVE research record

Public snapshot: October 24, 2026. Counts preserve their units: member updates, population updates, discarded replay calls, target presentations, tasks and condition-task outcomes are different measurements.

## Philosophy and method

The central hypothesis is that whole language-model members can develop useful contributions and learn to improve the predictions they make together. Shared frozen base weights make separate adapter members affordable to store; they do not remove the cost of running each member.

For the uniform K2 policy, both members condition on the same prefix and the next-token probability is `(p0 + p1) / 2`. Its supervised objective is the mean negative log of the correct target's mixture probability. Learned-gate variants use causal prefix information to mix members. Independent-training controls let the experiment distinguish collective-loss learning from separately trained adapters combined at inference.

The required evidence is generated-answer correctness, preserved capabilities and measured resources against meaningful controls. A smaller training loss, an oracle union of answers or a larger member count alone cannot answer the research question.

## Training milestones

| Stage | Completed work | Interpretation |
|---|---|---|
| Small-model engineering | 136 random-token optimizer updates across eight T4 configurations | Hardware and software calibration only |
| Expanded individual formation | 99 and 103 retained member updates; 1,654,784 target presentations | Partial completion of the original 128-update schedules; behavioral specialization unestablished |
| C32 collective SFT | 32 retained joint updates / 64 member steps; 262,144 shared targets from 737 records; one discarded replay | Both adapters changed under the collective objective |
| Matched training-loss diagnostic | Same 8,192 consumed target positions: NLL 0.788956 to 0.750893, 4.82% lower | Training fitting; post-run sample, not held-out accuracy |
| Collective RL attempts | No successful pretrained optimizer update | Unsuccessful attempts remain part of the resource history |
| Complete-answer Muon pilot | Eight retained collective updates; 32 examples presented twice | Four-question correctness stayed 3/4; completion improved on one previously unfinished answer |
| Learned-gate controls | Joint gate, independent gate and joint uniform each retained seven rounds; single control retained 21 updates | On six common reached questions, joint and independent both scored 2/6; ten primary and eight secondary questions were unobserved |

All three paired control arms retained 4,463 target presentations each; the single control retained 13,677. Shared update/data controls do not establish equal wall-clock compute. No Muon-versus-Adam superiority result is claimed.

## Generated-answer evaluations

The October14 K2 sample scored 6/12 on MATH-500 questions and 10/12 on HumanEval tasks. These small subsets are not full benchmark scores. Published scores from other models have different protocols and are not matched comparison runs.

The October16 characterization used twelve previously exposed tasks, greedy generation, a 512-token ceiling and a 480-second generation deadline. It captured 36 new answers and reused twelve unchanged joint answers.

| Condition | Maths | Coding | Total |
|---|---:|---:|---:|
| Member0 alone |4/6|6/6|10/12|
| Member1 alone |3/6|6/6|9/12|
| Independent K2 |2/6|5/6|7/12|
| Joint K2 |3/6|5/6|8/12|

The solo union solved ten tasks, all already solved by Member0. Joint added none and lost two solo successes. Its one graded win over independent is cutoff/extraction-sensitive: both derived the same mathematical value. This is not evidence of a better mathematical derivation. Collective code completion and repetition remain concrete bottlenecks.

A separate correction pilot tested draft, self-critique, peer-critique and direct revision on six tasks. All four routes scored 2/6; no wrong-to-right repair occurred. It produced 36 stages and 24 answer grades. This was frozen inference, not learned correction. A later supervised-revision probe passed bounded gradient checks but stopped at its cost gate with zero optimizer calls.

## Four further formation endpoints

Each fork started separately from the original joint checkpoint's Member0. Each successful run made three physical optimizer calls: two retained updates and one discarded exact replay. All exports were retained before the runtime was restored to its initial state.

| Curriculum | Targets/pass | Retained target presentations | Parent correct / wrong / unknown | Formed correct / wrong / unknown |
|---|---:|---:|---|---|
| Arithmetic |980|1,960|4 / 2 / 1 (7 cases)|5 / 1 / 1|
| Sequence/mapping |996|1,992|12 / 3 / 1 (16 cases)|12 / 3 / 1|
| Symbolic/discrete |884|1,768|12 / 3 / 1 (16 cases)|13 / 1 / 2|
| Tree sum/in-order |342|684|12 / 3 / 1 (16 cases)|12 / 3 / 1|

Every measured parent success was retained. No screen showed a comparable graded wrong-to-right repair. Arithmetic and symbolic completed the same previously unfinished factorial case, so their gains are not unique complementary successes. Unknowns remain in the denominator: the symbolic ship response is substantively wrong despite its extractor's unknown category. The tree probes were already correct before training and cover only partial transfer. Nine additional fixed-panel cases remain unmeasured for the arithmetic fork.

Four accepted trained adapter artifacts are a concrete milestone. They do not establish four complementary specialists or a K4 cooperation advantage. K4 numerical/recovery admission and matched joint-versus-independent learning remain unfinished in this snapshot.

## Engineering and resources

A real GPU trace found a full-sequence/cache discrepancy associated with different MoE expert choices near a routing tie. A diagnostic route intervention reversed the discrepancy; fixed arithmetic shapes then passed the recorded numerical guards. Production generation did not force expert routes. This is a tested numerical repair, not a claim that all model answers are correct.

Reported Kaggle phases used free two-T4 allocations and $0 cash. Provider allocation seconds are not summed physical-device seconds. Selected closed costs are:

| Phase | Provider allocation seconds |
|---|---:|
| Muon pilot, including failed predecessor |6,635.917|
| Learned-gate controls, including preceding failures/diagnostics |32,989.528155|
| Successful October14 benchmark allocation |9,095.520|
| Member characterization |9,312.244|
| Correction pilot |6,506.161|
| Revision-feasibility account observation |2,512.330|
| Stage2 parent baseline |2,999.188|
| Both arithmetic attempts |9,666.496|
| Sequence formation/evaluation |8,129.502|
| Symbolic formation/evaluation |7,331.692|
| Tree formation/evaluation |6,881.695|

This is not the project's lifetime total. Stable account deltas retain their attribution limitations; AI-service costs, complete historical local CPU, storage and human effort are not fully reconciled. Reused answers are not charged twice. Failed runs and discarded replays remain separate from retained learning.

## Bottlenecks and the next tests

1. **Complementarity:** existing screens show largely shared successes. Broader, qualified family coverage and per-member evaluation must establish distinct useful contributions.
2. **Completion and repetition:** preserve Member0's solved tasks when combining members, and measure output completion under declared time/token budgets.
3. **Cost and recovery:** admit actual four-member forward/backward/recovery costs before selecting a bounded collective-training run. Keep evaluation/export reserves and every failed attempt in the budget.
4. **Causal benefit:** compare formed-member joint learning with independent learning from the same initialization and declared exposures; include member and deployed-system outcomes.
5. **Generalization:** current samples are small and exposed. Historical pretraining exposure is unknown; no sealed final evaluation or larger-model parity result is claimed.

These are planned discriminating tests, not future results promised in advance.
