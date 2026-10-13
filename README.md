# RuACT supplementary code

This package provides one shared `verl` implementation for rubric-based reinforcement learning. RuACT combines RuACT-Teacher, which uses privileged rubric guidance for distillation, and RuACT-Critic, which refines reward rubrics. Experiment scripts select configurations of this shared implementation; they do not carry separate framework copies.

## Quick start

Install `requirements.txt` into an environment with mutually compatible CUDA, PyTorch, vLLM and FlashAttention installations. The submission does not pin dependency releases or include binary environments. Add the repository root and `scripts` to `PYTHONPATH` when importing method utilities directly.

Create a private `settings.local.json` by overriding the following fields from `experiments/settings.json`:

- `paths.model`: a local checkpoint matching the selected model profile.
- `paths.train_data` and `paths.validation_data`: prepared training and validation Parquet files.
- `paths.output`: an empty directory for a fresh experiment.
- `gpu_ids`: the intended visible GPU indices.
- `judge`: compatible chat-completions API base, model identifier and authentication.
- `generator`: the same fields for rubric generation, required only by RuACT-Critic.
- `paths.seed_library`: the frozen rubric candidate library for RuACT-Critic.

Endpoint credentials can instead be supplied through `HEALTHBENCH_JUDGE_API_KEY` and `HEALTHBENCH_GENERATOR_API_KEY`. Their corresponding `API_BASE` and `MODEL` environment variables also override settings. Do not redistribute populated settings files. The package has no default external service, credentials or hardware address.

```bash
bash experiments/main/medicine/4b/RuACT.sh --settings settings.local.json --dry-run
bash experiments/main/medicine/4b/RuACT.sh --settings settings.local.json
bash experiments/baselines/science/1.7b/Reweight.sh --settings settings.local.json
bash experiments/analysis/medicine/4b/no-filter.sh --settings settings.local.json
```

Dry runs construct commands without launching training, workers or API requests. All scripts call `experiments/train.py`. Configuration precedence is common settings, model/domain profile, private settings, method overrides, and finally explicit `hydra_overrides`. Set a method-specific field under `method_overrides` when overriding that method's defaults. Training is fresh by default, uses console logging, and refuses a nonempty output directory.

## File structure

```text
RuACT-submission/
├── README.md
├── THIRD_PARTY_NOTICES.txt
├── requirements.txt
├── experiments/
│   ├── train.py
│   ├── settings.json
│   ├── profiles/
│   │   ├── medicine-1.7b.json
│   │   ├── medicine-4b.json
│   │   ├── science-1.7b.json
│   │   └── science-4b.json
│   ├── main/{medicine,science}/{1.7b,4b}/
│   │   ├── GRPO.sh
│   │   ├── RuACT-Teacher.sh
│   │   ├── RuACT-Critic.sh
│   │   └── RuACT.sh
│   ├── baselines/{medicine,science}/{1.7b,4b}/
│   │   ├── GRPO.sh
│   │   ├── Reweight.sh
│   │   └── RuscaRL.sh
│   └── analysis/medicine/4b/
│       ├── GRPO-OPSD.sh
│       ├── DAPO-style.sh
│       ├── no-gate.sh
│       ├── no-filter.sh
│       ├── unreachable-RuACT-Teacher.sh
│       ├── saturated-RuACT-Critic.sh
│       ├── saturated-RuACT-Teacher.sh
│       └── unreachable-RuACT-Critic.sh
├── scripts/rubric_rl/
│   ├── verl_reward.py
│   ├── group_reward_loop.py
│   ├── reweight_reward.py
│   ├── simple_reweight.py
│   ├── rusca_reward.py
│   ├── ruscarl.py
│   ├── refinement_sidecar.py
│   └── refinement/
│       ├── protocol.py
│       ├── adapter.py
│       ├── registry.py
│       ├── worker.py
│       ├── single_pass_worker.py
│       └── bounded_retry.py
├── verl/
│   ├── trainer/ppo/
│   │   ├── ray_trainer.py
│   │   ├── distillation.py
│   │   └── dynamic_sampling.py
│   ├── experimental/agent_loop/
│   ├── workers/
│   └── utils/
├── configs/
├── evaluation/
└── tests/
```

Brace notation abbreviates actual directories. There are sixteen main scripts, twelve baseline scripts, and eight analysis scripts. The two GRPO categories invoke the same configuration. The remaining framework and utility files are omitted from the tree for readability.

## Training configurations

All four profiles target Qwen3 models of the indicated size and use the same Medicine-4B RuACT-Critic policy. Dataset paths are supplied separately, so choosing a profile does not silently select or download data.

| Setting | Shared default |
|---|---|
| Optimizer steps | 200 |
| Prompt groups per update | 64 |
| Responses per prompt | 8 |
| PPO mini-batch size | 32 |
| Actor/reference/log-probability micro-batch | 1 |
| Learning rate / warmup | 0.000001 / 12 steps |
| Prompt / response / model token limits | 1024 / 4096 / 6144 |
| Rollout temperature / top-p | 1 / 1 |
| Actor KL coefficient | 0.001, k1 estimator |
| KL in reward | Disabled |
| Seed | 42 |
| Checkpoint / validation frequency | Every 20 steps |
| RuACT-Teacher target coefficient | 0.1 for 1.7B; 0.05 for 4B |
| Internal teacher coefficient | Target divided by 8, with 50-step warmup |
| RuACT-Critic blend coefficient | 0.5 |
| Actor optimizer / reference parameters | CPU offload enabled |

Reweight and RuscaRL use actor micro-batches of four. RuscaRL additionally uses an 8192-token original prompt limit, a 12288-token total context, and rollout memory fraction 0.5. Its additional rubric prefix must fit the total context; overflow raises an error. The standard rollout memory fraction is 0.5. These differences are explicit in `method_overrides`; they are not hidden changes to the effective batch size.

The standard epoch limit is one and training is capped at 200 updates. The prepared dataset must supply enough batches for that cap. DAPO-style allows up to 200 dataset passes because rejected groups consume additional batches, while retaining the same cap of 200 optimizer updates. Insufficient valid groups or exhaustion of this limit raises an error.

## Shared RuACT-Critic policy

The Medicine-4B RuACT configuration is the common reference for all domains and sizes:

1. Consider positive-weight parent rubrics satisfied by all eight responses. Rank eligible parents by weight, retaining at most three parents per query and at most thirty-two queries per update.
2. Use a candidate budget of five per selected parent. Reuse up to five candidates with matching query, parent identity and parent fingerprint from the frozen seed library, then request exactly the missing count. Structural validation and deduplication can reduce the final pool; this does not trigger replenishment. The generator sees the query and parent requirement, not the sampled answers.
3. Candidate preparation has at most three total attempts, with one transport attempt per call, a forty-second call limit and a one-hundred-twenty-second parent budget. Exhaustion is an explicit failure, not a fabricated successful decomposition.
4. Judge the same eight responses against the merged candidate list. Keep candidates with pass counts from two through six inclusive, retaining the first three eligible candidates in candidate order. A parent with no eligible child retains its original reward contribution. There is no score-dependent regeneration to fill the three-child cap. Candidate acceptance here is structural validation; no independent semantic meta-judge is performed.
5. For an accepted parent, retain half its weight on the original rubric and distribute the other half equally among its retained children. Total parent-family weight is preserved.

Strict judge recovery preserves successful rows, retries failed rows and can split malformed or truncated responses into smaller rubric chunks. These scoring retries are separate from the three candidate-generation attempts. The trainer's result wait is bounded; endpoint failure cannot be treated as a valid all-zero scoring result in this path.

A frozen seed library is required by default. Selecting `allow_empty_seed_library: true` explicitly enables a different, empty-seed initialization; it must not be reported as reproducing a seeded run. The library and the exact input datasets are not bundled with this anonymous code package.

## Baselines and analyses

Reweight is an implementation inspired by rubric reweighting, not a complete reproduction of another training system. For binary occurrence score `s`, satisfaction is `g=s` for a positive rubric and `g=1-s` for a negative rubric. With group mean `p=mean(g)`, the multiplier is `1 + eta * 4*p*(1-p)`, with `eta=0.5`. Rewards average satisfaction using `abs(weight)` times this multiplier, normalized by the sum of these effective weights. Constant rubrics retain multiplier one and remain in the denominator. Its inherited judge failure policy makes up to five attempts and records explicit zero fallback on exhaustion; this differs from strict RuACT-Critic scoring and should be accounted for when comparing API failure rates.

RuscaRL adds a randomly selected subset of signed rubric descriptions to the generation prompt. For response index `i` in a group of size `K`, its reveal fraction is `(K-1-i)/(K-1) * sigmoid(-125*(progress-0.2))`, with `progress=(step-1)/(total_steps-1)`. The rounded subset size and original sampling rule are retained. Policy training uses the original unprivileged prompt, and validation has no rubric scaffold. The rollout backend is integrated into the shared asynchronous single-turn implementation; this consolidation is not a claim of a byte-identical historical runtime. RuscaRL's judge uses strict scoring with three total attempts.

| Analysis entry | Behavior |
|---|---|
| GRPO-OPSD | Inject all rubrics in original order, retain teacher verification, disable critic refinement, and allow a 16384-token privileged context. |
| DAPO-style | Drop groups whose reward standard deviation is at most 0.00000001 and refill the same update to sixty-four groups, within eight sampling rounds. It does not implement the complete DAPO algorithm. |
| no-gate | RuACT-Teacher with teacher verification disabled. |
| no-filter | RuACT-Critic with the child pass-count filter disabled; retain the first three structurally valid candidates from the same fixed candidate budget of five. |
| unreachable-RuACT-Teacher | Explicit alias for the normal teacher selector. |
| saturated-RuACT-Critic | Explicit alias for the normal critic selector. |
| saturated-RuACT-Teacher | Reverse the teacher selector while retaining other teacher settings. |
| unreachable-RuACT-Critic | Reverse the critic selector while retaining the common critic policy. |

For negative-weight teacher rubrics, the normal selector targets a violation present in all responses; the reversed selector targets a violation absent from all responses. Critic parent selection remains positive-weight only. The no-filter control preserves structural validation, deduplication, seed handling, scoring, and bounded candidate preparation. It removes the outcome filter rather than introducing another candidate-generation loop.

These are normalized submission recipes. Divergent historical worker implementations, infrastructure settings or jointly changed analysis settings are not silently preserved as additional framework forks. Historical numerical results should be associated with their actual original settings; the unified controls must not be described as retrospective reruns.

## Evaluation and verification

`evaluation/rollout_only.py` generates frozen responses. `evaluation/verify_eval.py` checks identity, coverage and scored outputs. `evaluation/aggregate_official.py` aggregates rubric-level scores. Evaluation data, checkpoints, endpoints and authentication are external inputs.

```bash
PYTHONPATH=.:scripts pytest -q tests
```

Tests cover the experiment matrix, resolved framework configurations, teacher selection and student-prefix behavior, fixed candidate budgets, seed reuse, retry bounds, strict scoring recovery, no-filter behavior, Reweight arithmetic, RuscaRL generation/training prompt separation, and dynamic sampling. These are local CPU checks with mocked remote calls. They do not establish end-to-end GPU training or reproduce benchmark scores.

## Privacy and third-party code

The package excludes credentials, private endpoints, server identifiers, training logs, checkpoints and private data. Tracking is disabled by default. Source comments and documentation strings have been removed; method descriptions are centralized in this English README. Necessary dependency compatibility checks and third-party license notices remain. `THIRD_PARTY_NOTICES.txt` preserves required attribution and copyright information.
