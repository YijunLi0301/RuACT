# Figure data and interpretation

`plot_results.py` reads the compact tables in `data/` and writes the README figures as SVG and PNG. Values are recovered experimental observations. Plotting performs no inference, scoring, smoothing, seed interpolation or data exclusion.

```bash
python analysis/plot_results.py
```

## Checkpoint score table

`checkpoint_scores.csv` records size, method, benchmark, cohort size, percent score, metric source, evaluator and the SHA-256 of the source summary or verification record. The underlying result artifacts were checked against their archived file digests before extraction. The public table excludes private machine paths, account names and run identifiers.

The evaluator is DeepSeek-V4.1-Flash with no thinking and temperature zero. Science RaR values come from verification records explicitly labelled with original rubric weights; their intermediate summary scores differ and are not used. Medicine-4B comes from archived summary records; HealthBench uses the raw mean, which equals the clipped mean in these particular runs. Other values are the recorded verified aggregates. The benchmark columns are not interchangeable aggregation protocols and no cross-benchmark average is inferred.

Science-1.7B GRPO is omitted: the archived label refers to a positive-rubric Teacher run. Science-4B Reweight is not available. Missing cells are not zeros. The consolidated launcher has not rerun these checkpoints, so these scores must not be described as validation of its exact current configuration.

## Teacher recovery

`teacher_recovery.csv` uses one fixed cohort of 931 initially unreachable criteria from 331 Medicine training queries. Eight responses are generated at each checkpoint. Recovery means at least one response passes that criterion at that checkpoint; it is not cumulative recovery and need not be monotone. Step-200 counts are 326 for GRPO, 361 for RuACT-Teacher and 399 for RuACT. The measured Teacher step-80 point is retained unchanged.

One RuACT step-80 judgment used an existing same-input local Qwen3-32B fallback. The curve is a descriptive diagnostic with no training-seed confidence interval. It does not establish out-of-distribution generalization.

## Critic dispersion and case

`critic_dispersion.csv` covers all 12,800 query groups per run, eight responses per group, steps 1–200. Each original/combined pair uses the same generated responses. Within-group standard deviation uses population normalization. The figure averages all groups in each nonoverlapping 20-step window, including groups where no child is applied.

The original/combined means are 0.1182018654 / 0.1316221940 for RuACT-Critic and 0.1217474121 / 0.1338927592 for RuACT. Early combined-run rewards were reconstructed from committed child scores; the maximum checked numerical discrepancy was below numerical precision. Increased spread alone is not evidence of better quality alignment.

`critic_case.json` preserves the recorded parent and child wording from one Medicine training group. All four pass counts refer to the same eight responses: parent 8/8, airway 2/8, cardiac monitoring 4/8, observation 6/8. The README shortens the descriptions only. This example illustrates discrimination; its medical statements are not an independently validated clinical protocol.

## Ablation comparisons

`ablation_scores.csv` contains complete step-200 Medicine-4B evaluation summaries, 2,242 RaR-Medicine and 500 HealthBench questions per entry. Each panel uses one evaluator. The gate panel uses Qwen3.7-Max; filter and target-choice panels use DeepSeek-V4.1-Flash. The Qwen gate result is nearly unchanged on RaR-Medicine; the difference is only about 0.0013 percentage points. Do not infer a meaningful RaR improvement from rounding.

The historical no-filter run also disabled regeneration and minimum-child handling, and had generator failures with retained historical seeds. It is a joint workflow comparison, not an isolated causal test of the outcome filter. Target-choice runs use matching evaluation cohorts, but full training-recipe matching is not established. The released analysis scripts now share the standard policy except for their explicitly selected control; that consolidation does not retroactively remove historical confounds.

An incomplete alternate-evaluator no-gate result is excluded. A separate guided/unguided recovery probe had mismatched prompts and is also excluded. No uncertainty bands or statistical significance are claimed from these single-run summaries.
