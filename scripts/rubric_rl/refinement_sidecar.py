from __future__ import annotations
import argparse
import asyncio
import errno
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_RUBRIC_ROOT = ROOT / "scripts"
for _path in (str(ROOT), str(_RUBRIC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from rubric_rl.atomic_io import atomic_replace_bytes, atomic_write_json_noreplace
from rubric_rl.refinement.protocol import SCHEMA_VERSION
from rubric_rl.refinement.worker import (
    WorkerConfig,
    configure_preverified_library,
    configure_generator_concurrency,
    configure_parent_concurrency,
    configure_judge_concurrency,
    process_refinement_job,
    validate_refinement_job,
)


def _mock_result(job):
    validate_refinement_job(job)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "job_id": job["job_id"],
        "transaction_id": job["transaction_id"],
        "query_id": job["query_id"],
        "source_query_version": int(job["query_version"]),
        "response_set_hash": job["response_set_hash"],
        "seed_bundle_hash": job["seed_bundle_hash"],
        "decompositions": [],
        "audit": {
            "backend": "mock",
            "selected_parent_count": len(job["selected_parents"]),
            "decomposition_count": 0,
            "generator_received_rollouts": False,
        },
    }


def _skipped_refinement_result(job, error, *, job_id, retry_count, backend):
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "job_id": job_id,
        "transaction_id": job["transaction_id"],
        "query_id": job["query_id"],
        "source_query_version": int(job["query_version"]),
        "response_set_hash": job["response_set_hash"],
        "seed_bundle_hash": job["seed_bundle_hash"],
        "decompositions": [],
        "audit": {
            "backend": backend,
            "refinement_skipped": True,
            "skip_reason": str(error)[:1200],
            "skip_after_attempts": int(retry_count),
            "selected_parent_count": len(job.get("selected_parents", [])),
            "decomposition_count": 0,
            "generator_received_rollouts": False,
        },
    }
    return result


def _terminal_error_result(job, error, *, job_id):
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "terminal_error",
        "job_id": job_id,
        "decompositions": [],
        "audit": {
            "terminal_error": {"error_type": type(error).__name__, "error": str(error)}
        },
    }
    if isinstance(job, dict):
        for field in (
            "transaction_id",
            "query_id",
            "response_set_hash",
            "seed_bundle_hash",
        ):
            if field in job:
                result[field] = job[field]
        if "query_version" in job:
            result["source_query_version"] = job["query_version"]
    return result


def _write_json_noreplace_with_retry(target, payload, retry_seconds, max_attempts):
    for attempt in range(1, max_attempts + 1):
        try:
            atomic_write_json_noreplace(target, payload)
            return
        except FileExistsError:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError(f"artifact collision at {target}")
            return
        except OSError as error:
            if error.errno not in (errno.ENOSPC, errno.EDQUOT):
                raise
            if attempt == max_attempts:
                raise
            print(
                json.dumps(
                    {
                        "event": "artifact_write_retry",
                        "path": str(target),
                        "error": str(error),
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "retry_seconds": retry_seconds,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(retry_seconds)


def _write_sidecar_status(
    path: Path, *, status: str, error: BaseException | None = None
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "pid": os.getpid(),
        "updated_at_unix": time.time(),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
    atomic_replace_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _heartbeat_loop(
    path: Path, interval: float, stop: threading.Event, errors: list[BaseException]
) -> None:
    while not stop.wait(interval):
        try:
            _write_sidecar_status(path, status="running")
        except BaseException as error:
            errors.append(error)
            return


async def _process_available(args, config) -> tuple[int, int]:
    suffix = f"v{SCHEMA_VERSION}"
    jobs_dir = args.state_dir / f"positive-refinement-jobs-{suffix}"
    results_dir = args.state_dir / f"positive-refinement-results-{suffix}"
    events_dir = args.state_dir / f"positive-refinement-events-{suffix}"
    for directory in (jobs_dir, results_dir, events_dir):
        directory.mkdir(parents=True, exist_ok=True)
    paths = [
        path
        for path in sorted(jobs_dir.glob("*.json"))
        if not (results_dir / path.name).exists()
    ][: args.max_jobs]

    async def process_one(path: Path) -> bool:
        result_path = results_dir / path.name
        if result_path.exists():
            return False
        job: dict = {}
        job_id = path.stem
        try:
            raw_job = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_job, dict):
                raise ValueError("job payload must be a JSON object")
            job = raw_job
            if str(job.get("job_id") or "") != job_id:
                raise ValueError("job filename and payload identity do not match")
            validate_refinement_job(job)
            if args.backend == "mock":
                result = _mock_result(job)
            else:
                retry = 0
                while True:
                    try:
                        result = await process_refinement_job(
                            job,
                            config=config,
                            parent_workers=args.parent_workers,
                            execution_mode=args.execution_mode,
                        )
                        break
                    except Exception as error:
                        retry += 1
                        skip_after = max(
                            1,
                            int(
                                os.environ.get(
                                    "POSITIVE_REFINEMENT_SKIP_AFTER_EMPTY_ATTEMPTS", "5"
                                )
                            ),
                        )
                        message = str(error)
                        candidate_exhaustion = (
                            "Refinement has no effective judged decomposition"
                            in message
                            or "generator must return between the requested minimum and maximum candidate count"
                            in message
                        )
                        if candidate_exhaustion and retry >= skip_after:
                            result = _skipped_refinement_result(
                                job,
                                error,
                                job_id=job_id,
                                retry_count=retry,
                                backend=args.backend,
                            )
                            print(
                                json.dumps(
                                    {
                                        "event": "refinement_refinement_skipped",
                                        "job_id": job_id,
                                        "query_id": job.get("query_id"),
                                        "attempts": retry,
                                        "reason": message[:500],
                                        "original_full_reward_preserved": True,
                                    },
                                    ensure_ascii=False,
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                            break
                        delay = min(60.0, 2.0 ** min(retry - 1, 5))
                        print(
                            json.dumps(
                                {
                                    "event": "refinement_job_retry",
                                    "job_id": job_id,
                                    "retry": retry,
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                    "sleep_seconds": delay,
                                },
                                ensure_ascii=False,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        await asyncio.sleep(delay)
        except Exception as error:
            diagnostic = events_dir / f"{path.stem}.terminal.json"
            _write_json_noreplace_with_retry(
                diagnostic,
                {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": path.stem,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "will_retry": False,
                },
                args.write_retry_seconds,
                args.max_write_attempts,
            )
            result = _terminal_error_result(job, error, job_id=job_id)
            terminal = True
        else:
            terminal = False
        _write_json_noreplace_with_retry(
            result_path, result, args.write_retry_seconds, args.max_write_attempts
        )
        return terminal

    if not paths:
        return (0, 0)
    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def run_bounded(path: Path) -> bool:
        async with semaphore:
            return await process_one(path)

    terminal_flags = await asyncio.gather(*(run_bounded(path) for path in paths))
    return (len(terminal_flags), sum(terminal_flags))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--preverified-library-path", type=Path, default=None)
    parser.add_argument("--backend", choices=("mock", "litellm"), default="litellm")
    parser.add_argument(
        "--base-url", default=os.environ.get("HEALTHBENCH_GENERATOR_API_BASE", "")
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get(
            "HEALTHBENCH_GENERATOR_API_KEY",
            os.environ.get("HEALTHBENCH_JUDGE_API_KEY", ""),
        ),
    )
    parser.add_argument(
        "--model", default=os.environ.get("HEALTHBENCH_GENERATOR_MODEL", "")
    )
    parser.add_argument("--meta-model", default=None)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("HEALTHBENCH_GENERATOR_TEMPERATURE", "0.8")),
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--no-topup", dest="topup_enabled", action="store_false", default=True
    )
    parser.add_argument(
        "--execution-mode",
        choices=("single_pass_batched",),
        default=os.environ.get(
            "POSITIVE_RUBRIC_REFINEMENT_EXECUTION_MODE", "single_pass_batched"
        ),
        help="single_pass_batched scores five candidates per parent in one query-wide checklist; no outcome-triggered top-up",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--max-jobs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--parent-workers", type=int, default=3)
    parser.add_argument("--parent-inflight", type=int, default=64)
    parser.add_argument("--generator-inflight", type=int, default=64)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--judge-inflight", type=int, default=64)
    parser.add_argument("--judge-attempts", type=int, default=3)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--write-retry-seconds", type=float, default=30.0)
    parser.add_argument("--max-write-attempts", type=int, default=3)
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    return parser.parse_args()


async def _watch_loop(
    args, config, heartbeat_errors: list[BaseException]
) -> tuple[int, int]:
    completed = errors = 0
    while True:
        if heartbeat_errors:
            raise RuntimeError("sidecar heartbeat writer failed") from heartbeat_errors[
                0
            ]
        if args.watch_pid is not None and (
            not Path(f"/proc/{args.watch_pid}").exists()
        ):
            break
        processed, terminal = await _process_available(args, config)
        completed += processed
        errors += terminal
        if terminal:
            raise RuntimeError(f"{terminal} refinement job(s) failed terminally")
        if not args.watch:
            break
        await asyncio.sleep(args.poll_seconds)
    return (completed, errors)


def main() -> int:
    args = parse_args()
    if args.execution_mode not in {"single_pass_batched"}:
        raise SystemExit("unsupported refinement execution mode")
    print(
        json.dumps(
            {
                "event": "refinement_execution_mode",
                "execution_mode": args.execution_mode,
                "bounded_retries": os.environ.get("POSITIVE_REFINEMENT_BOUNDED_RETRIES")
                == "1",
                "candidate_budget_per_parent": 5,
                "outcome_triggered_topup": False,
            }
        ),
        flush=True,
    )
    if (
        min(
            args.timeout,
            args.retry_backoff,
            args.poll_seconds,
            args.write_retry_seconds,
            args.heartbeat_seconds,
        )
        <= 0
        or min(
            args.max_jobs,
            args.workers,
            args.parent_workers,
            args.parent_inflight,
            args.generator_inflight,
            args.judge_workers,
            args.judge_inflight,
            args.judge_attempts,
            args.max_write_attempts,
        )
        <= 0
    ):
        raise SystemExit(
            "timeouts, backoff, job counts, and worker counts must be positive"
        )
    if args.parent_workers > 3:
        raise SystemExit(
            "parent-workers cannot exceed the per-query parent cap of three"
        )
    configure_generator_concurrency(args.generator_inflight)
    configure_parent_concurrency(args.parent_inflight)
    configure_judge_concurrency(args.judge_inflight)
    if args.preverified_library_path is not None:
        library_meta = configure_preverified_library(args.preverified_library_path)
        print(
            json.dumps(
                {"event": "preverified_library_loaded", **library_meta},
                ensure_ascii=False,
            ),
            flush=True,
        )
    suffix = f"v{SCHEMA_VERSION}"
    status_path = args.state_dir / f"positive-refinement-sidecar-{suffix}.json"
    heartbeat_stop = threading.Event()
    heartbeat_errors: list[BaseException] = []
    _write_sidecar_status(status_path, status="running")
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(status_path, args.heartbeat_seconds, heartbeat_stop, heartbeat_errors),
        name="positive-refinement-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    config = WorkerConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        meta_model=args.meta_model,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout,
        retry_backoff_seconds=args.retry_backoff,
        judge_workers=args.judge_workers,
        judge_attempts=args.judge_attempts,
        generator_concurrency=args.generator_inflight,
        topup_enabled=args.topup_enabled,
    )
    completed = errors = 0
    terminal_error: BaseException | None = None
    try:
        completed, errors = asyncio.run(_watch_loop(args, config, heartbeat_errors))
    except BaseException as error:
        terminal_error = error
        raise
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=max(args.heartbeat_seconds * 2, 1.0))
        _write_sidecar_status(
            status_path,
            status="terminal_error" if terminal_error is not None else "stopped",
            error=terminal_error,
        )
    print(json.dumps({"completed": completed, "errors": errors}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
