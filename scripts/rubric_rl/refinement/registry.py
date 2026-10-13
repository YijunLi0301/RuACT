from __future__ import annotations
import os
import hashlib
import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
from rubric_rl.atomic_io import atomic_replace_bytes
from .protocol import (
    BLEND_LAMBDA,
    REWARD_MODE,
    SCHEMA_VERSION,
    PositiveRefinementProtocolError,
    ordered_response_set_hash,
    parent_fingerprint,
    refinement_transaction_id,
    seed_bundle_hash,
)

META_CHECKS = (
    "factually_valid",
    "preserves_parent",
    "binary_judgeable",
    "relevant_to_query",
    "substantive",
)


def _initial_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_version": 0,
        "queries": {},
        "pending_jobs": {},
        "processed_jobs": {},
    }


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PositiveRefinementProtocolError(f"{field} must be a non-empty string")
    return value.strip()


class PositiveRefinementRegistry:

    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        suffix = f"v{SCHEMA_VERSION}"
        self.path = self.state_dir / f"positive-refinement-registry.{suffix}.json"
        self.jobs_dir = self.state_dir / f"positive-refinement-jobs-{suffix}"
        self.staged_jobs_dir = self.state_dir / f"positive-refinement-staged-{suffix}"
        self.results_dir = self.state_dir / f"positive-refinement-results-{suffix}"
        self.events_dir = self.state_dir / f"positive-refinement-events-{suffix}"
        self.sidecar_status_path = (
            self.state_dir / f"positive-refinement-sidecar-{suffix}.json"
        )
        for directory in (
            self.state_dir,
            self.jobs_dir,
            self.staged_jobs_dir,
            self.results_dir,
            self.events_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            state = _initial_state()
            self._write(state)
            return state
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
            raise PositiveRefinementProtocolError(
                f"completion decomposition requires a fresh v{SCHEMA_VERSION} registry"
            )
        if set(state) != set(_initial_state()):
            raise PositiveRefinementProtocolError(
                f"completion decomposition registry keys do not match v{SCHEMA_VERSION}"
            )
        if not all(
            (
                isinstance(state[key], dict)
                for key in ("queries", "pending_jobs", "processed_jobs")
            )
        ):
            raise PositiveRefinementProtocolError(
                "completion decomposition registry mappings are invalid"
            )
        return state

    def _write(self, state: Mapping[str, Any]) -> None:
        payload = (
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        atomic_replace_bytes(self.path, payload)

    def _commit(self, next_state: Mapping[str, Any]) -> None:
        materialized = deepcopy(dict(next_state))
        materialized["registry_version"] = int(self._state["registry_version"]) + 1
        self._write(materialized)
        self._state = materialized

    @property
    def registry_version(self) -> int:
        with self._lock:
            return int(self._state["registry_version"])

    def max_observed_step(self) -> int:
        with self._lock:
            values = [0]
            values.extend(
                (
                    int(item["global_step"])
                    for item in self._state["pending_jobs"].values()
                )
            )
            values.extend(
                (
                    int(item["global_step"])
                    for item in self._state["processed_jobs"].values()
                )
            )
            for query in self._state["queries"].values():
                values.extend(
                    (
                        int(item.get("created_step", 0))
                        for item in query["decompositions"].values()
                    )
                )
            return max(values)

    def snapshot(self, query_id: str) -> dict[str, Any]:
        query_id = _required_text(query_id, "query_id")
        with self._lock:
            query = self._state["queries"].get(query_id)
            return {
                "registry_version": self.registry_version,
                "query_version": int(query["query_version"]) if query else 0,
                "decompositions": deepcopy(
                    list((query or {"decompositions": {}})["decompositions"].values())
                ),
            }

    def blocked_query_ids(self) -> frozenset[str]:
        with self._lock:
            completed = {
                query_id
                for query_id, query in self._state["queries"].items()
                if query.get("decompositions")
            }
            return frozenset(set(self._state["pending_jobs"]) | completed)

    def reserve_job(self, job: Mapping[str, Any]) -> bool:
        query_id = _required_text(job.get("query_id"), "job.query_id")
        job_id = _required_text(job.get("job_id"), "job.job_id")
        parents = job.get("selected_parents")
        members = job.get("group_members")
        if not isinstance(parents, list) or not 1 <= len(parents) <= 3:
            raise PositiveRefinementProtocolError(
                "job must contain one to three selected parents"
            )
        if not isinstance(members, list) or len(members) != 8:
            raise PositiveRefinementProtocolError("job must freeze eight responses")
        if [item.get("rollout_index") for item in members] != list(range(8)):
            raise PositiveRefinementProtocolError(
                "job responses must be frozen in rollout order 0..7"
            )
        response_ids = [
            _required_text(item.get("id"), "response.id") for item in members
        ]
        if len(set(response_ids)) != 8:
            raise PositiveRefinementProtocolError("job response ids must be unique")
        response_set_hash = _required_text(
            job.get("response_set_hash"), "job.response_set_hash"
        )
        if ordered_response_set_hash(members) != response_set_hash:
            raise PositiveRefinementProtocolError(
                "job response set hash does not match its frozen responses"
            )
        fingerprints: list[str] = []
        pending_parents: list[dict[str, Any]] = []
        for parent in parents:
            rubric_id = _required_text(parent.get("rubric_id"), "parent.rubric_id")
            title = _required_text(parent.get("title"), "parent.title")
            description = _required_text(
                parent.get("description"), "parent.description"
            )
            weight = float(parent.get("weight", 0.0))
            supplied = _required_text(
                parent.get("parent_fingerprint"), "parent.parent_fingerprint"
            )
            valid_weight = weight > 0.0 or (
                bool(job.get("include_negative_weight")) and weight < 0.0
            )
            if not valid_weight or supplied != parent_fingerprint(
                rubric_id, title, description, weight
            ):
                raise PositiveRefinementProtocolError(
                    "job parent identity or weight polarity is invalid"
                )
            fingerprints.append(supplied)
            pending_parents.append(
                {
                    "parent_rubric_id": rubric_id,
                    "parent_fingerprint": supplied,
                    "parent_title": title,
                    "parent_description": description,
                    "parent_weight": weight,
                }
            )
        blend_lambda = float(job.get("blend_lambda", 0.0))
        supplied_seed_hash = _required_text(
            job.get("seed_bundle_hash"), "job.seed_bundle_hash"
        )
        if supplied_seed_hash != seed_bundle_hash(parents):
            raise PositiveRefinementProtocolError(
                "job seed bundle hash does not match its frozen candidates"
            )
        transaction_id = _required_text(job.get("transaction_id"), "job.transaction_id")
        expected = refinement_transaction_id(
            run_id=_required_text(job.get("run_id"), "job.run_id"),
            global_step=int(job.get("global_step", -1)),
            group_id=_required_text(job.get("group_id"), "job.group_id"),
            query_id=query_id,
            query_version=int(job.get("query_version", -1)),
            response_set_hash=response_set_hash,
            parent_fingerprints=fingerprints,
            seed_bundle_hash=supplied_seed_hash,
            blend_lambda=blend_lambda,
        )
        if job_id != transaction_id or job_id != expected:
            raise PositiveRefinementProtocolError(
                "job transaction identity does not match its immutable inputs"
            )
        with self._lock:
            if job_id in self._state["processed_jobs"]:
                return False
            if query_id in self._state["pending_jobs"]:
                return False
            query = self._state["queries"].get(query_id)
            query_version = int(query["query_version"]) if query else 0
            if query and query.get("decompositions"):
                return False
            if int(job.get("query_version", -1)) != query_version:
                raise PositiveRefinementProtocolError(
                    "job was built from a stale query snapshot"
                )
            next_state = deepcopy(self._state)
            next_state["queries"].setdefault(
                query_id, {"query_version": 0, "decompositions": {}}
            )
            next_state["pending_jobs"][query_id] = {
                "job_id": job_id,
                "transaction_id": transaction_id,
                "global_step": int(job["global_step"]),
                "source_query_version": query_version,
                "response_ids": response_ids,
                "response_set_hash": response_set_hash,
                "seed_bundle_hash": supplied_seed_hash,
                "blend_lambda": blend_lambda,
                "parents": pending_parents,
            }
            self._commit(next_state)
            return True

    def cancel_job(self, query_id: str, job_id: str) -> None:
        with self._lock:
            pending = self._state["pending_jobs"].get(query_id)
            if pending and pending["job_id"] == job_id:
                next_state = deepcopy(self._state)
                del next_state["pending_jobs"][query_id]
                self._commit(next_state)

    @staticmethod
    def _child_id(query_id: str, parent_id: str, title: str, description: str) -> str:
        raw = "\x00".join((query_id, parent_id, title, description))
        return f"{parent_id}:completion:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"

    def _validate_decomposition(
        self,
        decomposition: Mapping[str, Any],
        *,
        query_id: str,
        job_id: str,
        pending_parent: Mapping[str, Any],
        expected_ids: Sequence[str],
        created_step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        parent_id = _required_text(
            decomposition.get("parent_rubric_id"), "decomposition.parent_rubric_id"
        )
        if parent_id != pending_parent["parent_rubric_id"]:
            raise PositiveRefinementProtocolError(
                "decomposition does not match its reserved parent"
            )
        if (
            _required_text(
                decomposition.get("parent_fingerprint"),
                "decomposition.parent_fingerprint",
            )
            != pending_parent["parent_fingerprint"]
        ):
            raise PositiveRefinementProtocolError(
                "decomposition parent fingerprint is misaligned"
            )
        parent_weight = float(decomposition.get("parent_weight", 0.0))
        blend_lambda = float(decomposition.get("blend_lambda", 0.0))
        if (
            parent_weight != float(pending_parent["parent_weight"])
            or blend_lambda != BLEND_LAMBDA
        ):
            raise PositiveRefinementProtocolError(
                "decomposition parent weight or blend lambda is invalid"
            )
        children = decomposition.get("children")
        if not isinstance(children, list) or not 1 <= len(children) <= 3:
            raise PositiveRefinementProtocolError(
                "decomposition must retain one to three children"
            )
        raw_polarities = {str(child.get("polarity") or "") for child in children}
        unsupported_polarities = raw_polarities - {"positive", "negative", ""}
        if unsupported_polarities:
            raise PositiveRefinementProtocolError(
                "child polarity must be positive or negative when present"
            )
        if parent_weight > 0.0 and "negative" in raw_polarities:
            raise PositiveRefinementProtocolError(
                "a positive-weight parent cannot receive negative children"
            )
        if len(raw_polarities - {""}) > 1:
            raise PositiveRefinementProtocolError(
                "mixed-polarity children are unsupported"
            )
        if raw_polarities == {"negative"}:
            child_sign = -1.0
        elif raw_polarities == {"positive"}:
            child_sign = 1.0
        else:
            child_sign = 1.0 if parent_weight > 0.0 else -1.0
        records: list[dict[str, Any]] = []
        seen_completion_points: set[str] = set()
        child_weight = child_sign * abs(parent_weight) * blend_lambda / len(children)
        for child in children:
            title = _required_text(child.get("title"), "child.title")
            description = _required_text(child.get("description"), "child.description")
            completion_point = _required_text(
                child.get("completion_point"), "child.completion_point"
            )
            if description != completion_point:
                raise PositiveRefinementProtocolError(
                    "child description must be its atomic completion point"
                )
            necessity = str(child.get("necessity") or "optional").strip() or "optional"
            family = (
                str(child.get("semantic_family") or completion_point).strip()
                or completion_point
            )
            if necessity not in {"required", "optional"}:
                necessity = "optional"
            meta = child.get("meta_verification")
            if (
                not isinstance(meta, Mapping)
                or meta.get("decision") != "accept"
                or any((meta.get(check) is not True for check in META_CHECKS))
            ):
                raise PositiveRefinementProtocolError(
                    "child failed Meta Judge verification"
                )
            candidate_source = _required_text(
                child.get("candidate_source"), "child.candidate_source"
            )
            if candidate_source not in {"historical_seed", "online_generation"}:
                raise PositiveRefinementProtocolError(
                    "child candidate source is invalid"
                )
            source_provenance: dict[str, Any] = {
                "candidate_source": candidate_source,
                "source_candidate_index": int(child.get("source_candidate_index", -1)),
            }
            if source_provenance["source_candidate_index"] < 0:
                raise PositiveRefinementProtocolError(
                    "child source candidate index is invalid"
                )
            if candidate_source == "historical_seed":
                source_provenance.update(
                    {
                        "source_job_id": _required_text(
                            child.get("source_job_id"), "child.source_job_id"
                        ),
                        "source_step": int(child.get("source_step", 0)),
                        "source_pass_count": int(child.get("source_pass_count", -1)),
                    }
                )
            score_rows = child.get("judge_scores")
            if not isinstance(score_rows, list) or len(score_rows) != len(expected_ids):
                raise PositiveRefinementProtocolError(
                    "child Judge scores must cover all eight responses"
                )
            by_id: dict[str, int] = {}
            for row in score_rows:
                response_id = _required_text(
                    row.get("response_id"), "score.response_id"
                )
                score = row.get("score")
                if (
                    response_id in by_id
                    or isinstance(score, bool)
                    or score not in (0, 1)
                ):
                    raise PositiveRefinementProtocolError(
                        "child Judge scores are duplicated or non-binary"
                    )
                by_id[response_id] = int(score)
            if set(by_id) != set(expected_ids):
                raise PositiveRefinementProtocolError(
                    "child Judge response identities are misaligned"
                )
            pattern = tuple((by_id[response_id] for response_id in expected_ids))
            pass_count = sum(pattern)
            if (
                pass_count in {0, len(expected_ids)}
                and os.environ.get("RUACT_CRITIC_NO_FILTER") != "1"
            ):
                raise PositiveRefinementProtocolError(
                    "child Judge pattern must distinguish at least one response"
                )
            canonical_completion_point = " ".join(completion_point.casefold().split())
            if (
                not canonical_completion_point
                or canonical_completion_point in seen_completion_points
            ):
                raise PositiveRefinementProtocolError(
                    "child completion point is empty or duplicated"
                )
            if int(child.get("pass_count", -1)) != pass_count:
                raise PositiveRefinementProtocolError(
                    "child pass count does not match Judge scores"
                )
            seen_completion_points.add(canonical_completion_point)
            records.append(
                {
                    "rubric_id": self._child_id(
                        query_id, parent_id, title, description
                    ),
                    "title": title,
                    "description": description,
                    "completion_point": completion_point,
                    "necessity": necessity,
                    "semantic_family": family,
                    "pass_count": pass_count,
                    "pass_rate": pass_count / len(expected_ids),
                    "is_constant": pass_count in {0, len(expected_ids)},
                    "effective_weight": child_weight,
                    "judge_scores": deepcopy(score_rows),
                    "meta_verification": deepcopy(dict(meta)),
                    **source_provenance,
                }
            )
        record = {
            "parent_rubric_id": parent_id,
            "parent_fingerprint": str(pending_parent["parent_fingerprint"]),
            "parent_title": str(pending_parent["parent_title"]),
            "parent_description": str(pending_parent["parent_description"]),
            "parent_weight": parent_weight,
            "blend_lambda": blend_lambda,
            "parent_effective_weight": parent_weight * (1.0 - blend_lambda),
            "child_total_effective_weight": child_sign
            * abs(parent_weight)
            * blend_lambda,
            "children": [
                {
                    key: value
                    for key, value in child.items()
                    if key not in {"judge_scores", "meta_verification"}
                }
                for child in records
            ],
            "reward_mode": REWARD_MODE,
            "source": "parent_preserved_multi_child_refinement",
            "created_step": int(created_step),
            "source_job_id": job_id,
        }
        return (record, {**deepcopy(record), "children": records})

    def _validate_result_locked(
        self, result: Mapping[str, Any], *, created_step: int
    ) -> dict[str, Any]:
        if result.get("schema_version") != SCHEMA_VERSION:
            raise PositiveRefinementProtocolError(
                "unsupported completion decomposition result schema"
            )
        job_id = _required_text(result.get("job_id"), "result.job_id")
        if result.get("status") != "ok":
            terminal = (result.get("audit") or {}).get("terminal_error") or {}
            error_type = str(terminal.get("error_type") or "unknown")
            error = str(terminal.get("error") or "sidecar job failed")
            raise PositiveRefinementProtocolError(
                f"sidecar returned terminal job failure ({error_type}): {error}"
            )
        query_id = _required_text(result.get("query_id"), "result.query_id")
        if job_id in self._state["processed_jobs"]:
            raise PositiveRefinementProtocolError("result job was already committed")
        pending = self._state["pending_jobs"].get(query_id)
        if not pending or pending["job_id"] != job_id:
            raise PositiveRefinementProtocolError("result has no matching pending job")
        if int(created_step) != int(pending["global_step"]):
            raise PositiveRefinementProtocolError(
                "result step does not match its source transaction"
            )
        if (
            _required_text(result.get("transaction_id"), "result.transaction_id")
            != pending["transaction_id"]
        ):
            raise PositiveRefinementProtocolError(
                "result transaction identity is misaligned"
            )
        if (
            _required_text(result.get("response_set_hash"), "result.response_set_hash")
            != pending["response_set_hash"]
        ):
            raise PositiveRefinementProtocolError(
                "result response set hash is misaligned"
            )
        if (
            _required_text(result.get("seed_bundle_hash"), "result.seed_bundle_hash")
            != pending["seed_bundle_hash"]
        ):
            raise PositiveRefinementProtocolError(
                "result seed bundle hash is misaligned"
            )
        query = self._state["queries"].get(query_id)
        if query is None or int(result.get("source_query_version", -1)) != int(
            query["query_version"]
        ):
            raise PositiveRefinementProtocolError(
                "result was built from a stale query version"
            )
        decompositions = result.get("decompositions")
        if not isinstance(decompositions, list):
            raise PositiveRefinementProtocolError(
                "result.decompositions must be a list"
            )
        pending_by_parent = {
            item["parent_rubric_id"]: item for item in pending["parents"]
        }
        parent_ids = [
            _required_text(item.get("parent_rubric_id"), "parent_rubric_id")
            for item in decompositions
        ]
        if len(parent_ids) != len(set(parent_ids)) or set(parent_ids) - set(
            pending_by_parent
        ):
            raise PositiveRefinementProtocolError(
                "result contains duplicate or unreserved parents"
            )
        staged = [
            self._validate_decomposition(
                decomposition,
                query_id=query_id,
                job_id=job_id,
                pending_parent=pending_by_parent[
                    str(decomposition["parent_rubric_id"])
                ],
                expected_ids=pending["response_ids"],
                created_step=created_step,
            )
            for decomposition in decompositions
        ]
        return {
            "status": "validated",
            "job_id": job_id,
            "query_id": query_id,
            "pending": deepcopy(pending),
            "staged": staged,
            "applied_decompositions": [deepcopy(active) for _, active in staged],
            "selected_parent_count": len(pending["parents"]),
            "decomposition_count": len(staged),
            "selected_child_count": sum(
                (len(active["children"]) for _, active in staged)
            ),
        }

    def validate_result(
        self, result: Mapping[str, Any], *, created_step: int
    ) -> dict[str, Any]:
        with self._lock:
            return self._validate_result_locked(result, created_step=created_step)

    def commit_result(
        self, result: Mapping[str, Any], *, created_step: int
    ) -> dict[str, Any]:
        with self._lock:
            summary = self._validate_result_locked(result, created_step=created_step)
            job_id = summary["job_id"]
            query_id = summary["query_id"]
            pending = summary["pending"]
            staged = summary["staged"]
            next_state = deepcopy(self._state)
            next_query = next_state["queries"][query_id]
            for record, _ in staged:
                next_query["decompositions"][record["parent_rubric_id"]] = deepcopy(
                    record
                )
            if staged:
                next_query["query_version"] = int(next_query["query_version"]) + 1
            del next_state["pending_jobs"][query_id]
            next_state["processed_jobs"][job_id] = {
                "global_step": int(created_step),
                "query_id": query_id,
                "transaction_id": str(pending["transaction_id"]),
                "decomposition_parent_ids": [
                    record["parent_rubric_id"] for record, _ in staged
                ],
            }
            self._commit(next_state)
            return {**summary, "status": "committed", "pending": None, "staged": None}

    def synchronize_to_checkpoint(
        self, checkpoint_step: int, *, require_empty: bool = False
    ) -> dict[str, int]:
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < 0
        ):
            raise PositiveRefinementProtocolError(
                "checkpoint step must be a non-negative integer"
            )
        with self._lock:
            if require_empty and any(
                (
                    self._state["queries"],
                    self._state["pending_jobs"],
                    self._state["processed_jobs"],
                )
            ):
                raise PositiveRefinementProtocolError(
                    "fresh training requires an empty refinement registry"
                )
            next_state = deepcopy(self._state)
            removed_pending = removed_processed = removed_decompositions = 0
            for query_id, pending in list(next_state["pending_jobs"].items()):
                pending_step = int(pending.get("global_step", -1))
                if pending_step <= checkpoint_step:
                    raise PositiveRefinementProtocolError(
                        f"pending refinement for {query_id} is not newer than checkpoint step {checkpoint_step}"
                    )
                del next_state["pending_jobs"][query_id]
                removed_pending += 1
            for job_id, processed in list(next_state["processed_jobs"].items()):
                processed_step = int(processed.get("global_step", -1))
                if processed_step < 0:
                    raise PositiveRefinementProtocolError(
                        "processed refinement has an invalid step"
                    )
                if processed_step > checkpoint_step:
                    del next_state["processed_jobs"][job_id]
                    removed_processed += 1
            for query_id, query in list(next_state["queries"].items()):
                decompositions = query.get("decompositions")
                if not isinstance(decompositions, dict):
                    raise PositiveRefinementProtocolError(
                        "query decomposition mapping is invalid"
                    )
                retained_source_jobs: set[str] = set()
                for parent_id, decomposition in list(decompositions.items()):
                    created_step = int(decomposition.get("created_step", -1))
                    source_job_id = _required_text(
                        decomposition.get("source_job_id"), "source_job_id"
                    )
                    if created_step > checkpoint_step:
                        processed = self._state["processed_jobs"].get(source_job_id)
                        if (
                            not processed
                            or int(processed.get("global_step", -1)) != created_step
                            or str(processed.get("query_id") or "") != query_id
                        ):
                            raise PositiveRefinementProtocolError(
                                "ahead decomposition cannot be tied to an exact processed transaction"
                            )
                        del decompositions[parent_id]
                        removed_decompositions += 1
                        continue
                    processed = next_state["processed_jobs"].get(source_job_id)
                    if (
                        not processed
                        or int(processed.get("global_step", -1)) != created_step
                        or str(processed.get("query_id") or "") != query_id
                        or (
                            parent_id
                            not in processed.get("decomposition_parent_ids", [])
                        )
                    ):
                        raise PositiveRefinementProtocolError(
                            "retained decomposition has no exact committed transaction"
                        )
                    retained_source_jobs.add(source_job_id)
                if len(retained_source_jobs) > 1:
                    raise PositiveRefinementProtocolError(
                        "query contains decompositions from multiple transactions"
                    )
                query["query_version"] = len(retained_source_jobs)
                if (
                    not decompositions
                    and query["query_version"] == 0
                    and (query_id not in next_state["pending_jobs"])
                ):
                    del next_state["queries"][query_id]
            for job_id, processed in next_state["processed_jobs"].items():
                query_id = _required_text(
                    processed.get("query_id"), "processed.query_id"
                )
                parent_ids = processed.get("decomposition_parent_ids")
                if not isinstance(parent_ids, list) or len(parent_ids) != len(
                    set(parent_ids)
                ):
                    raise PositiveRefinementProtocolError(
                        "processed decomposition parent ids are invalid"
                    )
                query = next_state["queries"].get(query_id)
                decompositions = (query or {}).get("decompositions", {})
                for parent_id in parent_ids:
                    decomposition = decompositions.get(parent_id)
                    if (
                        not decomposition
                        or decomposition.get("source_job_id") != job_id
                    ):
                        raise PositiveRefinementProtocolError(
                            "processed transaction is missing its committed decomposition"
                        )
            changed = bool(
                removed_pending or removed_processed or removed_decompositions
            )
            if changed:
                self._commit(next_state)
            return {
                "removed_pending": removed_pending,
                "removed_processed": removed_processed,
                "removed_decompositions": removed_decompositions,
            }

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)
