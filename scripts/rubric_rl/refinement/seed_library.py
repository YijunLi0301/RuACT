from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from .protocol import PositiveRefinementProtocolError, parent_fingerprint

SEED_LIBRARY_SCHEMA_VERSION = 1


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PositiveRefinementProtocolError(f"{field} must be a non-empty string")
    return value.strip()


def seed_lookup_key(
    query_id: str, rubric_id: str, fingerprint: str
) -> tuple[str, str, str]:
    return (
        _required_text(query_id, "query_id"),
        _required_text(rubric_id, "rubric_id"),
        _required_text(fingerprint, "parent_fingerprint"),
    )


class PositiveRefinementSeedLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise PositiveRefinementProtocolError(
                f"seed library is missing: {self.path}"
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("library_schema") != "refinement-seed-library"
        ):
            raise PositiveRefinementProtocolError(
                "unsupported positive-refinement seed library schema"
            )
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise PositiveRefinementProtocolError("seed library entries must be a list")
        entries: dict[tuple[str, str, str], tuple[dict[str, Any], ...]] = {}
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise PositiveRefinementProtocolError(
                    "seed library entry must be an object"
                )
            query_id = _required_text(raw_entry.get("query_id"), "seed.query_id")
            rubric_id = _required_text(
                raw_entry.get("parent_rubric_id"), "seed.parent_rubric_id"
            )
            title = _required_text(raw_entry.get("parent_title"), "seed.parent_title")
            description = _required_text(
                raw_entry.get("parent_description"), "seed.parent_description"
            )
            weight = float(raw_entry.get("parent_weight", 0.0))
            fingerprint = _required_text(
                raw_entry.get("parent_fingerprint"), "seed.parent_fingerprint"
            )
            if weight == 0.0 or fingerprint != parent_fingerprint(
                rubric_id, title, description, weight
            ):
                raise PositiveRefinementProtocolError(
                    "seed entry does not match its root parent"
                )
            raw_candidates = raw_entry.get("children") or raw_entry.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise PositiveRefinementProtocolError(
                    "seed entry must contain children/candidates"
                )
            candidates: list[dict[str, Any]] = []
            seen_points: set[str] = set()
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping):
                    raise PositiveRefinementProtocolError(
                        "seed candidate must be an object"
                    )
                point = _required_text(
                    raw_candidate.get("completion_point"), "seed.completion_point"
                )
                canonical = " ".join(point.casefold().split())
                if canonical in seen_points:
                    raise PositiveRefinementProtocolError(
                        "seed entry contains duplicate completion points"
                    )
                seen_points.add(canonical)
                candidates.append(
                    {
                        "title": _required_text(
                            raw_candidate.get("title"), "seed.title"
                        ),
                        "completion_point": point,
                        "source_job_id": str(
                            raw_candidate.get("source_job_id")
                            or f"preverified:{query_id}:{rubric_id}:{len(candidates)}"
                        ),
                        "source_step": int(
                            raw_candidate.get(
                                "source_step", raw_entry.get("created_step", 0)
                            )
                        ),
                        "source_pass_count": int(
                            raw_candidate.get("source_pass_count", -1)
                        ),
                        **(
                            {"polarity": str(raw_candidate["polarity"])}
                            if raw_candidate.get("polarity") in {"positive", "negative"}
                            else {}
                        ),
                    }
                )
            key = seed_lookup_key(query_id, rubric_id, fingerprint)
            if key in entries:
                raise PositiveRefinementProtocolError(
                    "seed library contains a duplicate exact parent key"
                )
            entries[key] = tuple(candidates)
        self._entries = entries
        self.metadata = dict(payload.get("metadata") or {})

    def lookup(
        self,
        *,
        query_id: str,
        rubric_id: str,
        title: str,
        description: str,
        weight: float,
        limit: int,
    ) -> tuple[dict[str, Any], ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise PositiveRefinementProtocolError(
                "seed lookup limit must be a positive integer"
            )
        fingerprint = parent_fingerprint(rubric_id, title, description, weight)
        values = self._entries.get(
            seed_lookup_key(query_id, rubric_id, fingerprint), ()
        )
        return tuple((dict(item) for item in values[:limit]))

    def as_lookup_mapping(
        self, *, limit: int
    ) -> Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise PositiveRefinementProtocolError(
                "seed lookup limit must be a positive integer"
            )
        return {
            key: tuple((dict(item) for item in values[:limit]))
            for key, values in self._entries.items()
        }

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "PositiveRefinementSeedLibrary",
    "SEED_LIBRARY_SCHEMA_VERSION",
    "seed_lookup_key",
]
