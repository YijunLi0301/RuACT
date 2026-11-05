# Training

## Configuration

All shell entrypoints call `experiments/train.py`. Settings are applied in this order: `configs/training.json`, `configs/methods/critic.json`, the selected domain/size profile, the private settings file, method-specific training overrides, and explicit `hydra_overrides`. The final command is written to the effective trainer configuration with secrets removed.

Set `paths.model`, `paths.train_data`, `paths.validation_data`, `paths.output`, `gpu_ids`, and the `judge` connection fields. Critic-enabled runs also require the `generator` fields and a frozen `paths.seed_library`. `allow_empty_seed_library: true` explicitly selects an alternative initialization; it is not equivalent to a seeded run.

Environment overrides are `HEALTHBENCH_JUDGE_API_BASE`, `HEALTHBENCH_JUDGE_API_KEY`, `HEALTHBENCH_JUDGE_MODEL`, and the corresponding `HEALTHBENCH_GENERATOR_*` variables. Use a compatible chat-completions service. Model and service names are inputs; the code selects no external provider. Keep populated settings and credentials private.

The launcher uses fresh training, console logging and an empty output directory. It supervises the Critic sidecar and stops the trainer if that worker fails. This launcher does not pretend that loading an inference-only model is a full optimizer resume.

## Teacher policy

Teacher and combined RuACT select the same unreachable rubrics, with the same verification gate and student-prefix distillation. Positive rubrics are unreachable when none of eight responses passes; for negative-weight rubrics, all responses must exhibit the unwanted behavior. Selection retains at most three criteria. The privileged teacher sees the selected guidance, while policy inputs remain unprivileged. The verification gate requires an improvement over the student group. Accepted guidance distills the privileged distribution along the selected student's response, not the teacher's generated continuation.

The target coefficient is 0.1 at 1.7B and 0.05 at 4B, represented internally as target / 8, with a 50-step warmup. Teacher probabilities are detached from gradient computation. `no-gate` disables verification. `saturated-RuACT-Teacher` reverses the parent selector. `GRPO-OPSD` injects all rubrics in original order and allows a 16,384-token privileged context.

## Critic policy

The standalone Critic and combined RuACT read the same `critic` configuration:

| Field | Standard value | Meaning |
|---|---|---|
| `execution_mode` | `feedback_batched` | Judge candidate pools jointly, then regenerate only for parents below the minimum |
| `candidate_count` | 5 | Fixed initial pool size, including matching frozen seeds |
| `min_children_per_parent` | 1 | Trigger one extra proposal round if fewer survive |
| `max_children_per_parent` | 3 | Retention cap, not a target to refill indefinitely |
| `topup_round_limit` | 1 | At most one additional candidate batch |
| `max_queries_per_step` | 32 | Query budget per optimizer update |
| `semantic_meta_enabled` | false | Independent semantic validation is opt-in |
| `meta_model` | null | Use the generator model unless explicitly overridden |

Select at most three saturated positive parents per query. Reuse up to five frozen candidates only when query ID, parent ID and parent fingerprint match. Generate the missing initial candidates; validate structure and deduplicate them. Score candidates on the same eight responses and retain those passed by 2–6 responses inclusive, up to three per parent.

If fewer than the minimum survive, the generator receives the query, parent and rejected candidate wording for one additional proposal batch. It does not receive response texts or pass-count statistics. New candidates are checked, deduplicated against the initial pool, scored and combined with earlier surviving children. Exhausting this round does not trigger a third round or an attempt to fill the cap. Any previously valid children are preserved even if the additional round cannot reach the configured minimum. Parents with no valid child keep their original contribution.

`topup_round_limit: 0` disables feedback regeneration. `execution_mode: single_pass_batched` is an explicit single-pass control. Neither is the standard recipe. `no-filter` disables only the pass-count selection in the consolidated recipe; other method settings remain shared.

With semantic validation enabled, a separate structured judgment checks factual validity, preservation of the parent, binary judgeability, query relevance and substantive content. The stage processes seeds as well as newly proposed candidates. A disabled stage is recorded as disabled, not accepted; it must not be described as an executed meta-judge.

For parent weight `w`, the accepted parent retains `(1-lambda)w`; the retained children share `lambda*w` equally, with `lambda=0.5`. This preserves each parent-family weight. The bounds are three parents per query and three children per parent, not three children in total across a query.

Generation transport/format retries and feedback regeneration are separate budgets. Candidate preparation has at most three attempts within a 120-second parent budget. Strict judge recovery can retry malformed rows and split rubric lists; the trainer bounds its result wait. Unavailable scoring is not a valid all-zero judgment. Sidecar records distinguish retries, semantic decisions, candidate rounds and final retention.

## Baselines

**Reweight.** For binary occurrence score `s`, satisfaction is `g=s` for a positive rubric and `g=1-s` for a negative rubric. With `p=mean(g)` over the group, multiply the absolute rubric weight by `1 + eta*4*p*(1-p)`, with `eta=0.5`. Normalize by the sum of effective weights. Constant rubrics retain multiplier one and remain in the denominator. This is a reweighting baseline inspired by POW3R, not a full reproduction. Its inherited scorer records zero fallback after five failed attempts; account for such failures when comparing runs.

**RuscaRL.** For response index `i` in a group of size `K`, reveal a randomly selected rubric subset at fraction `(K-1-i)/(K-1) * sigmoid(-125*(progress-0.2))`. Here `progress=(step-1)/(total_steps-1)`. Generation uses the scaffold; policy training uses the original public prompt, and validation has no scaffold. Context overflow fails explicitly.

**DAPO-style.** Discard groups with reward standard deviation at most `1e-8` and refill to the same effective batch size within eight sampling rounds. This is the dynamic group-filtering control, not the entire DAPO algorithm.

## Training budget and integration

The common cap is 200 optimizer updates, 64 prompt groups per update, eight responses per prompt, PPO mini-batch 32, learning rate `1e-6` with 12-step warmup, and actor KL `0.001` with the k1 estimator. Reward KL is disabled. Save and validate every 20 steps. The dataset must contain enough batches for the configured epoch budget; the launcher does not silently cycle an exhausted dataset. DAPO-style permits additional dataset passes while retaining the update cap.

Actor, reference and log-probability micro-batches default to one. Reweight and RuscaRL use actor micro-batch four. RuscaRL has an 8,192-token prompt limit and 12,288-token context; common limits are 1,024 and 6,144. Rollout responses are capped at 4,096 tokens. GPU IDs are explicitly supplied by the operator.

`ruact/teacher/` holds method planning and gate logic. Distributed scheduling remains in `verl/trainer/ppo/ray_trainer.py`; student-prefix forwards and loss integration remain in `verl/workers/actor/`. `ruact/critic/` owns the refinement transaction protocol and sidecar. `ruact/group_reward_loop.py` connects grouped judgments to the rollout runtime. All experiments use these same modules.
