from __future__ import annotations
import json
import os
import time
from copy import deepcopy
from itertools import combinations
from typing import Any, Mapping
import numpy as np
from rubric_rl.atomic_io import atomic_write_json_noreplace
from .protocol import (
    PositiveRefinementConfig,
    PositiveRefinementProtocolError,
    build_refinement_jobs,
    parent_fingerprint,
)
from .registry import PositiveRefinementRegistry
from .seed_library import PositiveRefinementSeedLibrary


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PositiveRefinementProtocolError(f"{field} must be a mapping")
    return value


def _row(values: Any, index: int, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


class PositiveRubricRefinementAdapter:
    def __init__(self, config: PositiveRefinementConfig, *, run_id: str):
        self.config = config
        self.run_id = str(run_id).strip() or "unnamed-run"
        self.registry = (
            PositiveRefinementRegistry(config.state_dir) if config.enabled else None
        )
        self.seed_library = (
            PositiveRefinementSeedLibrary(config.seed_library_path)
            if config.enabled and config.seed_library_path is not None
            else None
        )
        self._seed_lookup = (
            self.seed_library.as_lookup_mapping(limit=config.seed_candidate_cap)
            if self.seed_library is not None
            else {}
        )
        self._processed = (
            set(self.registry.export_state()["processed_jobs"])
            if self.registry
            else set()
        )
        self._staged_results: list[dict[str, Any]] = []

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | Any | None, *, run_id: str):
        return cls(PositiveRefinementConfig.from_mapping(config), run_id=run_id)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @staticmethod
    def _query_id(reward_model: Any) -> str:
        model = _mapping(reward_model, "reward_model")
        truth = _mapping(model.get("ground_truth"), "reward_model.ground_truth")
        query_id = str(truth.get("prompt_id") or "").strip()
        if not query_id:
            raise PositiveRefinementProtocolError(
                "completion decomposition requires ground_truth.prompt_id"
            )
        return query_id

    def _overlay_registry(self, batch: Any) -> dict[str, float]:
        assert self.registry is not None
        reward_models = batch.non_tensor_batch["reward_model"]
        raw_extra_infos = batch.non_tensor_batch.get("extra_info")
        new_models = np.empty(len(batch), dtype=object)
        new_extra_infos = np.empty(len(batch), dtype=object)
        overlay_queries: set[str] = set()
        child_count = 0
        for index in range(len(batch)):
            reward_model = deepcopy(
                dict(_mapping(reward_models[index], "reward_model"))
            )
            truth = deepcopy(
                dict(_mapping(reward_model.get("ground_truth"), "ground_truth"))
            )
            query_id = self._query_id(reward_model)
            snapshot = self.registry.snapshot(query_id)
            by_parent = {
                str(item["parent_rubric_id"]): item
                for item in snapshot["decompositions"]
            }
            rubrics: list[dict[str, Any]] = []
            applied: set[str] = set()
            raw_rubrics = truth.get("rubrics")
            if raw_rubrics is None:
                raw_rubrics = ()
            elif isinstance(raw_rubrics, np.ndarray):
                raw_rubrics = raw_rubrics.tolist()
            if not isinstance(raw_rubrics, (list, tuple)):
                raise PositiveRefinementProtocolError(
                    "ground_truth.rubrics must be a sequence"
                )
            for raw_rubric in raw_rubrics:
                rubric = deepcopy(dict(_mapping(raw_rubric, "rubric")))
                rubric_id = str(rubric.get("rubric_key") or "")
                decomposition = by_parent.get(rubric_id)
                if decomposition is None:
                    rubrics.append(rubric)
                    continue
                original_weight = float(decomposition["parent_weight"])
                if float(rubric.get("weight", 0.0)) != original_weight or (
                    original_weight <= 0.0 and (not self.config.include_negative_weight)
                ):
                    raise PositiveRefinementProtocolError(
                        "active parent no longer matches its decomposition"
                    )
                live_fingerprint = parent_fingerprint(
                    rubric_id,
                    str(rubric.get("title") or ""),
                    str(rubric.get("description") or ""),
                    original_weight,
                )
                if live_fingerprint != str(decomposition["parent_fingerprint"]):
                    raise PositiveRefinementProtocolError(
                        "active parent semantics changed after refinement"
                    )
                children = list(decomposition["children"])
                if not children:
                    raise PositiveRefinementProtocolError(
                        "active decomposition has no completion points"
                    )
                blend_lambda = float(decomposition["blend_lambda"])
                parent_weight = original_weight * (1.0 - blend_lambda)
                child_weight = original_weight * blend_lambda / len(children)
                parent = deepcopy(rubric)
                parent["weight"] = parent_weight
                parent["dcd_role"] = "parent"
                parent["dcd_original_weight"] = original_weight
                parent["dcd_blend_lambda"] = blend_lambda
                rubrics.append(parent)
                for child in children:
                    if self.config.include_negative_weight:
                        child_weight = float(
                            child.get(
                                "effective_weight",
                                original_weight * blend_lambda / len(children),
                            )
                        )
                    rubrics.append(
                        {
                            "rubric_key": str(child["rubric_id"]),
                            "title": str(child["title"]),
                            "description": str(child["description"]),
                            "weight": child_weight,
                            "source": "dynamic_rubric",
                            "dcd_parent_rubric_key": rubric_id,
                            "dcd_necessity": str(child["necessity"]),
                            "dcd_role": "child",
                            "dcd_original_parent_weight": original_weight,
                            "dcd_blend_lambda": blend_lambda,
                        }
                    )
                applied.add(rubric_id)
                child_count += len(children)
            if set(by_parent) != applied:
                raise PositiveRefinementProtocolError(
                    "decomposition parent is absent from current query rubrics"
                )
            if applied:
                overlay_queries.add(query_id)
            truth["rubrics"] = rubrics
            truth["rubric_snapshot_version"] = int(snapshot["registry_version"])
            truth["rubric_query_version"] = int(snapshot["query_version"])
            reward_model["ground_truth"] = truth
            new_models[index] = reward_model
            extra = deepcopy(dict(_row(raw_extra_infos, index, {}) or {}))
            extra["rubric_snapshot_version"] = int(snapshot["registry_version"])
            extra["rubric_query_version"] = int(snapshot["query_version"])
            new_extra_infos[index] = extra
        batch.non_tensor_batch["reward_model"] = new_models
        batch.non_tensor_batch["extra_info"] = new_extra_infos
        return {
            "dcd/overlay_query_count": float(len(overlay_queries)),
            "dcd/overlay_child_count": float(child_count),
        }

    def _assert_clean_step_boundary(self) -> None:
        assert self.registry is not None
        if self._staged_results:
            raise PositiveRefinementProtocolError(
                "previous refinement results were not finalized after actor update"
            )
        if self.registry.export_state()["pending_jobs"]:
            raise PositiveRefinementProtocolError(
                "pending refinement state reached a new batch without checkpoint synchronization"
            )

    def synchronize_to_checkpoint(
        self, checkpoint_step: int, *, require_empty: bool
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        assert self.registry is not None
        if self._staged_results:
            raise PositiveRefinementProtocolError(
                "cannot synchronize while refinement results are staged"
            )
        summary = self.registry.synchronize_to_checkpoint(
            checkpoint_step, require_empty=require_empty
        )
        self._processed = set(self.registry.export_state()["processed_jobs"])
        return {
            "dcd/resume_removed_pending_count": float(summary["removed_pending"]),
            "dcd/resume_removed_processed_count": float(summary["removed_processed"]),
            "dcd/resume_removed_decomposition_count": float(
                summary["removed_decompositions"]
            ),
        }

    def prepare_batch(
        self, batch: Any, *, global_step: int
    ) -> tuple[Any, dict[str, float]]:
        del global_step
        if not self.enabled:
            return (batch, {})
        assert self.registry is not None
        self._assert_clean_step_boundary()
        metrics = self._overlay_registry(batch)
        metrics["dcd/registry_version"] = float(self.registry.registry_version)
        return (batch, metrics)

    @staticmethod
    def _observation_rows(batch: Any) -> list[dict[str, Any]]:
        data = batch.non_tensor_batch
        required = {
            "uid",
            "rollout_index",
            "prompt_id",
            "query",
            "rubric_scores",
            "rubric_ids",
            "rubric_titles",
            "rubric_descriptions",
            "rubric_weights",
            "candidate_final_answer",
            "occurrence_id",
        }
        if not required.issubset(data):
            raise PositiveRefinementProtocolError(
                f"DCD observation is missing {sorted(required - set(data))}"
            )
        return [
            {
                "group_id": str(data["uid"][index]),
                "query_id": str(data["prompt_id"][index]),
                "query": str(data["query"][index]),
                "original_index": index,
                "rollout_index": int(data["rollout_index"][index]),
                "response_id": str(data["occurrence_id"][index]),
                "response_text": str(data["candidate_final_answer"][index]),
                "rubric_scores": list(data["rubric_scores"][index]),
                "rubric_ids": list(data["rubric_ids"][index]),
                "rubric_titles": list(data["rubric_titles"][index]),
                "rubric_descriptions": list(data["rubric_descriptions"][index]),
                "rubric_weights": list(data["rubric_weights"][index]),
                "rubric_query_version": int(
                    _row(data.get("rubric_query_version"), index, 0)
                ),
            }
            for index in range(len(batch))
        ]

    def _stage_and_publish(self, job: Mapping[str, Any]) -> bool:
        assert self.registry is not None
        job_id = str(job["job_id"])
        staged = self.registry.staged_jobs_dir / f"{job_id}.json"
        published = self.registry.jobs_dir / f"{job_id}.json"
        if staged.exists():
            if json.loads(staged.read_text(encoding="utf-8")) != job:
                raise PositiveRefinementProtocolError(
                    "staged job identity collides with different content"
                )
        else:
            atomic_write_json_noreplace(staged, job)
        if not self.registry.reserve_job(job):
            return False
        try:
            if published.exists():
                if json.loads(published.read_text(encoding="utf-8")) != job:
                    raise PositiveRefinementProtocolError(
                        "published job identity collides with different content"
                    )
            else:
                atomic_write_json_noreplace(published, job)
        except Exception:
            self.registry.cancel_job(str(job["query_id"]), job_id)
            raise
        return True

    def queue_same_group_rubrics(self, batch: Any, *, global_step: int):
        if not self.enabled:
            return ([], {})
        assert self.registry is not None
        jobs = build_refinement_jobs(
            self._observation_rows(batch),
            run_id=self.run_id,
            global_step=global_step,
            expected_group_size=self.config.expected_group_size,
            max_queries=self.config.max_queries_per_step,
            max_parents_per_query=self.config.max_parents_per_query,
            min_candidate_count=self.config.min_candidate_count,
            max_candidate_count=self.config.max_candidate_count,
            min_children_per_parent=self.config.min_children_per_parent,
            max_children_per_parent=self.config.max_children_per_parent,
            zero_std_epsilon=self.config.zero_std_epsilon,
            blend_lambda=self.config.blend_lambda,
            include_negative_weight=self.config.include_negative_weight,
            seed_candidates_by_parent=self._seed_lookup,
            seed_candidate_cap=self.config.seed_candidate_cap,
            blocked_query_ids=self.registry.blocked_query_ids(),
        )
        queued = [job for job in jobs if self._stage_and_publish(job)]
        preverified_hits = sum(
            (
                len(parent.get("seed_candidates") or ())
                for job in queued
                for parent in job["selected_parents"]
            )
        )
        return (
            queued,
            {
                "dcd/triggered_query_count": float(len(jobs)),
                "dcd/queued_query_count": float(len(queued)),
                "dcd/selected_parent_count": float(
                    sum((len(job["selected_parents"]) for job in queued))
                ),
                "dcd/saturated_positive_count": float(
                    sum((int(job["positive_saturated_count"]) for job in queued))
                ),
                "dcd/saturated_positive_weight_fraction_mean": float(
                    sum(
                        (
                            float(job["saturated_positive_weight_fraction"])
                            for job in queued
                        )
                    )
                    / len(queued)
                    if queued
                    else 0.0
                ),
                "dcd/seed_candidate_count": float(preverified_hits),
                "dcd/preverified_hit_count": float(preverified_hits),
                "dcd/preverified_parent_hit_count": float(
                    sum(
                        (
                            bool(parent.get("seed_candidates"))
                            for job in queued
                            for parent in job["selected_parents"]
                        )
                    )
                ),
                "dcd/zero_std_candidate_count": float(
                    sum((bool(job.get("zero_std_gate")) for job in queued))
                ),
                "dcd/zero_std_group_count": float(
                    sum((bool(job.get("zero_std_gate")) for job in queued))
                ),
                "dcd/seed_parent_hit_count": float(
                    sum(
                        (
                            bool(parent.get("seed_candidates"))
                            for job in queued
                            for parent in job["selected_parents"]
                        )
                    )
                ),
                "dcd/negative_saturated_count": float(
                    sum((int(job.get("negative_saturated_count", 0)) for job in queued))
                ),
                "dcd/saturated_signed_weight_fraction_mean": float(
                    sum(
                        (
                            float(
                                job.get(
                                    "saturated_weight_fraction",
                                    job["saturated_positive_weight_fraction"],
                                )
                            )
                            for job in queued
                        )
                    )
                    / len(queued)
                    if queued
                    else 0.0
                ),
            },
        )

    def _assert_sidecar_live(self) -> None:
        assert self.registry is not None
        status_path = self.registry.sidecar_status_path
        if not status_path.is_file():
            raise PositiveRefinementProtocolError(
                "positive-refinement sidecar status file is missing"
            )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PositiveRefinementProtocolError(
                f"positive-refinement sidecar status is unreadable: {type(error).__name__}"
            ) from error
        if (
            status.get("schema_version")
            != self.registry.export_state()["schema_version"]
        ):
            raise PositiveRefinementProtocolError(
                "positive-refinement sidecar schema is misaligned"
            )
        if status.get("status") != "running":
            raise PositiveRefinementProtocolError(
                f"positive-refinement sidecar is not running: {status.get('status')!r}"
            )
        updated_at = status.get("updated_at_unix")
        pid = status.get("pid")
        if (
            isinstance(updated_at, bool)
            or not isinstance(updated_at, (int, float))
            or time.time() - float(updated_at)
            > self.config.sidecar_heartbeat_timeout_seconds
        ):
            raise PositiveRefinementProtocolError(
                "positive-refinement sidecar heartbeat is stale"
            )
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise PositiveRefinementProtocolError(
                "positive-refinement sidecar process is not alive"
            )
        try:
            os.kill(pid, 0)
        except OSError as error:
            raise PositiveRefinementProtocolError(
                "positive-refinement sidecar process is not alive"
            ) from error

    def _wait_for_result(self, job: Mapping[str, Any]) -> dict[str, Any]:
        assert self.registry is not None
        job_id = str(job["job_id"])
        result_path = self.registry.results_dir / f"{job_id}.json"
        deadline = time.monotonic() + self.config.result_wait_timeout_seconds
        while True:
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise PositiveRefinementProtocolError(
                        f"sidecar result is unreadable: {type(error).__name__}"
                    ) from error
                if str(result.get("job_id") or "") != job_id:
                    raise PositiveRefinementProtocolError(
                        "sidecar result identity is misaligned"
                    )
                return result
            if (
                time.monotonic() >= deadline
                and os.environ.get("Refinement_JUDGE_RETRY_UNTIL_SUCCESS") != "1"
            ):
                raise PositiveRefinementProtocolError(
                    f"timed out waiting for positive-refinement result {job_id}"
                )
            self._assert_sidecar_live()
            time.sleep(self.config.result_poll_seconds)

    @staticmethod
    def _scalar_reward(
        ids: list[str], scores: list[int], weights: list[float]
    ) -> float:
        del ids
        denominator = sum((weight for weight in weights if weight > 0.0))
        if denominator <= 0.0:
            raise PositiveRefinementProtocolError(
                "reward requires at least one positive rubric"
            )
        return (
            sum((score * weight for score, weight in zip(scores, weights, strict=True)))
            / denominator
        )

    def _apply_decompositions(
        self,
        batch: Any,
        reward_tensor: Any,
        reward_extra: dict[str, Any],
        *,
        job: Mapping[str, Any],
        decompositions: list[Mapping[str, Any]],
    ) -> tuple[Any, list[float], list[float], list[float], list[float]]:
        if not decompositions:
            zeros = [0.0] * len(batch)
            return (reward_tensor, zeros, zeros, zeros, zeros)
        updated = reward_tensor.clone()
        response_mask = batch.batch["response_mask"]
        expected_ids = {str(item["id"]) for item in job["group_members"]}
        occurrence_ids = reward_extra["occurrence_id"]
        query_ids = reward_extra["prompt_id"]
        indexed: list[tuple[Mapping[str, Any], list[dict[str, int]]]] = []
        for decomposition in decompositions:
            child_maps = []
            for child in decomposition["children"]:
                by_id = {
                    str(item["response_id"]): int(item["score"])
                    for item in child["judge_scores"]
                }
                if set(by_id) != expected_ids:
                    raise PositiveRefinementProtocolError(
                        "completion scores do not cover the source group"
                    )
                child_maps.append(by_id)
            indexed.append((decomposition, child_maps))
        deltas = [0.0] * len(batch)
        original_rewards = [0.0] * len(batch)
        combined_rewards = [0.0] * len(batch)
        coverages = [0.0] * len(batch)
        applied: set[str] = set()
        for index in range(len(batch)):
            response_id = str(_row(occurrence_ids, index, ""))
            if response_id not in expected_ids:
                continue
            if str(_row(query_ids, index, "")) != str(job["query_id"]):
                raise PositiveRefinementProtocolError(
                    "source response has the wrong query id"
                )
            applied.add(response_id)
            ids = list(_row(reward_extra.get("rubric_ids"), index, []) or [])
            scores = [
                int(value)
                for value in list(
                    _row(reward_extra.get("rubric_scores"), index, []) or []
                )
            ]
            weights = [
                float(value)
                for value in list(
                    _row(reward_extra.get("rubric_weights"), index, []) or []
                )
            ]
            if not len(ids) == len(scores) == len(weights):
                raise PositiveRefinementProtocolError(
                    "static rubric evidence is misaligned"
                )
            denominator = sum((weight for weight in weights if weight > 0.0))
            original_reward = self._scalar_reward(ids, scores, weights)
            numerator_delta = 0.0
            row_coverages: list[float] = []
            parent_ids: list[str] = []
            child_ids: list[str] = []
            for decomposition, child_maps in indexed:
                parent_id = str(decomposition["parent_rubric_id"])
                if parent_id not in ids:
                    raise PositiveRefinementProtocolError(
                        "selected parent is absent from current reward evidence"
                    )
                position = ids.index(parent_id)
                weight = float(decomposition["parent_weight"])
                if weight != weights[position] or (
                    weight <= 0.0 and (not self.config.include_negative_weight)
                ):
                    raise PositiveRefinementProtocolError(
                        "parent weight changed before same-group DCD"
                    )
                coverage = float(np.mean([by_id[response_id] for by_id in child_maps]))
                parent_score = scores[position]
                blend_lambda = float(decomposition["blend_lambda"])
                numerator_delta += blend_lambda * weight * (coverage - parent_score)
                row_coverages.append(coverage)
                parent_ids.append(parent_id)
                child_ids.extend(
                    (str(child["rubric_id"]) for child in decomposition["children"])
                )
            delta = numerator_delta / denominator
            combined = original_reward + delta
            deltas[index] = delta
            original_rewards[index] = original_reward
            combined_rewards[index] = combined
            coverages[index] = float(np.mean(row_coverages)) if row_coverages else 0.0
            valid_positions = response_mask[index].nonzero(as_tuple=False).flatten()
            if int(valid_positions.numel()) == 0:
                raise PositiveRefinementProtocolError(
                    "cannot attach reward to an empty response"
                )
            updated[index, int(valid_positions[-1].item())] += delta
            diagnostics = {
                "dcd_parent_ids": parent_ids,
                "dcd_child_ids": child_ids,
                "dcd_original_reward": original_reward,
                "dcd_completion_coverage": coverages[index],
                "dcd_parent_share": 1.0 - float(indexed[0][0]["blend_lambda"]),
                "dcd_child_share": float(indexed[0][0]["blend_lambda"]),
                "dcd_combined_reward": combined,
                "dcd_reward_delta": delta,
            }
            for key, value in diagnostics.items():
                rows = reward_extra.setdefault(key, [None for _ in range(len(batch))])
                rows[index] = value
                if key in batch.non_tensor_batch:
                    batch.non_tensor_batch[key][index] = value
        if applied != expected_ids:
            raise PositiveRefinementProtocolError(
                "same-group DCD missed a source response"
            )
        return (updated, deltas, original_rewards, combined_rewards, coverages)

    @staticmethod
    def _ordering_metrics(
        before: list[float], after: list[float]
    ) -> tuple[int, int, int]:
        reversals = comparable = sign_flips = 0
        mean_before = float(np.mean(before))
        mean_after = float(np.mean(after))
        for left, right in combinations(range(len(before)), 2):
            old_diff = before[left] - before[right]
            new_diff = after[left] - after[right]
            if old_diff != 0.0:
                comparable += 1
                reversals += int(old_diff * new_diff < 0.0)
        for old, new in zip(before, after, strict=True):
            sign_flips += int((old - mean_before) * (new - mean_after) < 0.0)
        return (reversals, comparable, sign_flips)

    def apply_queued_same_group_rubrics(
        self,
        batch: Any,
        reward_tensor: Any,
        reward_extra_infos: dict[str, Any],
        *,
        queued: list[dict[str, Any]],
        global_step: int,
    ):
        if not self.enabled:
            if queued:
                raise PositiveRefinementProtocolError(
                    "disabled DCD received queued jobs"
                )
            return (reward_tensor, {})
        assert self.registry is not None
        if not queued:
            return (
                reward_tensor,
                {
                    "dcd/decomposition_count": 0.0,
                    "dcd/selected_child_count": 0.0,
                    "dcd/preverified_hit_count": 0.0,
                    "dcd/preverified_complete_reuse_count": 0.0,
                    "dcd/preverified_topup_count": 0.0,
                    "dcd/zero_std_group_count": 0.0,
                    "dcd/zero_std_to_nonzero_count": 0.0,
                    "dcd/wait_seconds": 0.0,
                },
            )
        if self._staged_results:
            raise PositiveRefinementProtocolError(
                "refinement results are already staged for actor update"
            )
        started = time.monotonic()
        updated = reward_tensor
        all_deltas: list[float] = []
        all_original: list[float] = []
        all_combined: list[float] = []
        all_coverages: list[float] = []
        std_before: list[float] = []
        std_after: list[float] = []
        decomposition_count = child_count = insufficient_children = meta_rejected = 0
        zero_std_queued = zero_std_recovered = 0
        generator_calls = generator_prompt_characters = generator_prompt_bytes = (
            seed_pre_meta_skipped
        ) = 0
        preverified_complete_reuse = preverified_topup = preverified_topup_failure = 0
        constant_zero = constant_full = child_weak = child_balanced = 0
        reversals = comparable_pairs = advantage_sign_flips = 0
        for job in queued:
            zero_std_queued += int(bool(job.get("zero_std_gate")))
            result = self._wait_for_result(job)
            if result.get("status") == "terminal_error":
                terminal = (result.get("audit") or {}).get("terminal_error") or {}
                raise PositiveRefinementProtocolError(
                    f"positive-refinement sidecar job failed terminally: {terminal.get('error_type', 'unknown')}: {terminal.get('error', '')}"
                )
            for parent in (result.get("audit") or {}).get("parents") or []:
                if str(parent.get("status") or "") == "terminal_parent_failure":
                    raise PositiveRefinementProtocolError(
                        "positive-refinement result contains a terminal parent failure"
                    )
            effective_result = deepcopy(result)
            result_audit = result.get("audit") or {}
            generator_calls += int(result_audit.get("generator_call_count", 0))
            generator_prompt_characters += int(
                result_audit.get("generator_prompt_character_count", 0)
            )
            generator_prompt_bytes += int(
                result_audit.get("generator_prompt_utf8_bytes", 0)
            )
            seed_pre_meta_skipped += int(
                result_audit.get("seed_pre_meta_skipped_count", 0)
            )
            if self.config.mode == "shadow":
                effective_result["decompositions"] = []
            summary = self.registry.validate_result(
                effective_result, created_step=global_step
            )
            self._staged_results.append(
                {
                    "job_id": str(job["job_id"]),
                    "global_step": int(global_step),
                    "result": effective_result,
                }
            )
            decompositions = list(summary.get("applied_decompositions") or [])
            if decompositions and self.config.mode == "active":
                updated, deltas, original, combined, coverages = (
                    self._apply_decompositions(
                        batch,
                        updated,
                        reward_extra_infos,
                        job=job,
                        decompositions=decompositions,
                    )
                )
                member_indices = [
                    index
                    for index, value in enumerate(reward_extra_infos["occurrence_id"])
                    if str(value) in {str(item["id"]) for item in job["group_members"]}
                ]
                group_before = [original[index] for index in member_indices]
                group_after = [combined[index] for index in member_indices]
                std_before.append(float(np.std(group_before)))
                std_after.append(float(np.std(group_after)))
                zero_std_recovered += int(
                    bool(job.get("zero_std_gate"))
                    and std_before[-1] <= 1e-08
                    and (std_after[-1] > 1e-08)
                )
                rev, comp, flips = self._ordering_metrics(group_before, group_after)
                reversals += rev
                comparable_pairs += comp
                advantage_sign_flips += flips
                all_deltas.extend((deltas[index] for index in member_indices))
                all_original.extend(group_before)
                all_combined.extend(group_after)
                all_coverages.extend((coverages[index] for index in member_indices))
            decomposition_count += len(decompositions)
            child_count += sum((len(item["children"]) for item in decompositions))
            for parent in (result.get("audit") or {}).get("parents") or []:
                seed_count = int(parent.get("seed_candidate_count", 0))
                generated_count = int(parent.get("generated_candidate_count", 0))
                preverified_complete_reuse += int(
                    seed_count > 0 and generated_count == 0
                )
                preverified_topup += int(seed_count > 0 and generated_count > 0)
                preverified_topup_failure += int(
                    seed_count > 0
                    and generated_count > 0
                    and (str(parent.get("status")) != "selected")
                )
                status = str(parent.get("status") or "")
                insufficient_children += int(
                    status == "insufficient_meta_valid_children"
                )
                meta_rejected += int(status == "meta_rejected_all")
                for candidate in parent.get("candidate_pass_counts") or []:
                    pass_count = int(candidate.get("pass_count", -1))
                    constant_zero += int(pass_count == 0)
                    constant_full += int(pass_count == self.config.expected_group_size)
                    child_weak += int(
                        pass_count in (1, self.config.expected_group_size - 1)
                    )
                    child_balanced += int(
                        2 <= pass_count <= self.config.expected_group_size - 2
                    )
        return (
            updated,
            {
                "dcd/decomposition_count": float(decomposition_count),
                "dcd/selected_child_count": float(child_count),
                "dcd/terminal_parent_failure_count": 0.0,
                "dcd/terminal_job_failure_count": 0.0,
                "dcd/insufficient_children_parent_count": float(insufficient_children),
                "dcd/meta_rejected_parent_count": float(meta_rejected),
                "dcd/generator_call_count": float(generator_calls),
                "dcd/generator_prompt_character_count": float(
                    generator_prompt_characters
                ),
                "dcd/generator_prompt_utf8_bytes": float(generator_prompt_bytes),
                "dcd/seed_pre_meta_skipped_count": float(seed_pre_meta_skipped),
                "dcd/preverified_complete_reuse_count": float(
                    preverified_complete_reuse
                ),
                "dcd/preverified_topup_count": float(preverified_topup),
                "dcd/preverified_topup_failure_count": float(preverified_topup_failure),
                "dcd/child_zero_of_eight_count": float(constant_zero),
                "dcd/child_eight_of_eight_count": float(constant_full),
                "dcd/child_weak_count": float(child_weak),
                "dcd/balanced_child_rate": float(
                    child_balanced
                    / (constant_zero + constant_full + child_weak + child_balanced)
                    if constant_zero + constant_full + child_weak + child_balanced
                    else 0.0
                ),
                "dcd/wait_seconds": float(time.monotonic() - started),
                "dcd/original_reward_mean": (
                    float(np.mean(all_original)) if all_original else 0.0
                ),
                "dcd/completion_coverage_mean": (
                    float(np.mean(all_coverages)) if all_coverages else 0.0
                ),
                "dcd/combined_reward_mean": (
                    float(np.mean(all_combined)) if all_combined else 0.0
                ),
                "dcd/reward_delta_mean": (
                    float(np.mean(all_deltas)) if all_deltas else 0.0
                ),
                "dcd/reward_delta_std": (
                    float(np.std(all_deltas)) if all_deltas else 0.0
                ),
                "dcd/group_std_before_mean": (
                    float(np.mean(std_before)) if std_before else 0.0
                ),
                "dcd/group_std_after_mean": (
                    float(np.mean(std_after)) if std_after else 0.0
                ),
                "dcd/group_std_delta_mean": (
                    float(np.mean(np.subtract(std_after, std_before)))
                    if std_before
                    else 0.0
                ),
                "dcd/zero_std_to_nonzero_count": float(
                    sum(
                        (
                            before == 0.0 and after > 0.0
                            for before, after in zip(std_before, std_after, strict=True)
                        )
                    )
                ),
                "dcd/zero_std_group_count": float(zero_std_queued),
                "dcd/zero_std_recovery_rate": (
                    float(zero_std_recovered / zero_std_queued)
                    if zero_std_queued
                    else 0.0
                ),
                "dcd/order_reversal_rate": (
                    float(reversals / comparable_pairs) if comparable_pairs else 0.0
                ),
                "dcd/advantage_sign_flip_rate": (
                    float(advantage_sign_flips / len(all_original))
                    if all_original
                    else 0.0
                ),
            },
        )

    def finalize_after_actor_update(self, *, global_step: int) -> dict[str, float]:
        if not self.enabled:
            return {}
        assert self.registry is not None
        committed = 0
        while self._staged_results:
            staged = self._staged_results[0]
            if int(staged["global_step"]) != int(global_step):
                raise PositiveRefinementProtocolError(
                    "staged refinement result belongs to a different actor step"
                )
            self.registry.commit_result(staged["result"], created_step=global_step)
            self._processed.add(str(staged["job_id"]))
            self._staged_results.pop(0)
            committed += 1
        return {
            "dcd/committed_job_count": float(committed),
            "dcd/registry_version": float(self.registry.registry_version),
        }

    def apply_same_group_rubrics(
        self,
        batch: Any,
        reward_tensor: Any,
        reward_extra_infos: dict[str, Any],
        *,
        global_step: int,
    ):
        queued, metrics = self.queue_same_group_rubrics(batch, global_step=global_step)
        updated, apply_metrics = self.apply_queued_same_group_rubrics(
            batch,
            reward_tensor,
            reward_extra_infos,
            queued=queued,
            global_step=global_step,
        )
        metrics.update(apply_metrics)
        return (updated, metrics)
