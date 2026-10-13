from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL = (
    Path(__file__).resolve().parent / "data" / "medcine-1.7b" / "official_test.parquet"
)
N_ROWS = int(os.environ.get("OFFICIAL_N_ROWS", "2242"))
NEG_TOTAL = int(os.environ.get("OFFICIAL_NEG_TOTAL", "3051"))
OFFICIAL_WEIGHTS_SOURCE = os.environ.get(
    "OFFICIAL_WEIGHTS_SOURCE",
    "RaR-Medicine test split explicit integer weights, including negative weights",
)


def load_official(path: Path):
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if len(rows) != N_ROWS:
        raise SystemExit(f"official parquet rows={len(rows)} != {N_ROWS}")
    weights, desc_sha, neg = ([], [], 0)
    for row in rows:
        rs = row["rubric"]
        if not rs:
            raise SystemExit("official row with empty rubric list")
        ws = []
        for r in rs:
            w = r.get("weight")
            if w is not None and (not isinstance(w, (int, float))):
                raise SystemExit(f"non-numeric official weight {w!r}")
            ws.append(w)
            if w is not None and w < 0:
                neg += 1
        weights.append(ws)
        digest = hashlib.sha1(
            "\n".join((str(r["description"]) for r in rs)).encode()
        ).hexdigest()
        desc_sha.append(digest)
    if neg != NEG_TOTAL:
        raise SystemExit(f"official neg count={neg} != {NEG_TOTAL}")
    return (weights, desc_sha, neg)


def load_results(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != N_ROWS:
        raise SystemExit(f"results rows={len(rows)} != {N_ROWS}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--official", type=Path, default=OFFICIAL)
    args = ap.parse_args()
    weights, desc_sha, neg_total = load_official(args.official)
    rows = load_results(args.results)
    per = []
    neg_hit = 0
    desc_bad = []
    den0 = 0
    seen: set[int] = set()
    row_index_mode = os.environ.get("OFFICIAL_ROW_INDEX", "pid")
    for rowpos, row in enumerate(rows):
        pid = str(row["prompt_id"])
        if row_index_mode == "position":
            idx = rowpos
        else:
            try:
                idx = int(pid.split("-")[3])
            except (ValueError, IndexError) as err:
                raise SystemExit(f"unparsable prompt_id {pid!r}") from err
        if idx in seen:
            raise SystemExit(f"duplicate idx {idx}")
        seen.add(idx)
        if not 0 <= idx < N_ROWS:
            raise SystemExit(f"idx {idx} out of range")
        digest = hashlib.sha1(
            "\n".join((str(rb["description"]) for rb in row["rubrics"])).encode()
        ).hexdigest()
        if digest != desc_sha[idx]:
            desc_bad.append(idx)
            continue
        num = den = 0.0
        for s, w in zip(row["rubric_scores"], weights[idx]):
            if w is None:
                continue
            num += float(s) * w
            if w > 0:
                den += w
            if w < 0 and int(s) == 1:
                neg_hit += 1
        if den <= 0:
            den0 += 1
            continue
        per.append(num / den)
    if desc_bad:
        raise SystemExit(
            f"desc sha mismatch on {len(desc_bad)} rows, e.g. idx {desc_bad[:5]}"
        )
    if len(seen) != N_ROWS:
        raise SystemExit(f"idx coverage {len(seen)} != {N_ROWS}")
    extra = {
        "score_mean_original": statistics.fmean(per) if per else None,
        "score_median_original": statistics.median(per) if per else None,
        "score_std_population_original": (
            statistics.pstdev(per) if len(per) > 1 else 0.0
        ),
        "neg_total": neg_total,
        "neg_hit": neg_hit,
        "neg_hit_rate": neg_hit / neg_total if neg_total else None,
        "official_caliber": {
            "weights_source": OFFICIAL_WEIGHTS_SOURCE,
            "desc_mismatch_rows": 0,
            "skipped_zero_den_rows": den0,
            "neg_count_verified": neg_total,
        },
        "official_verified_at": datetime.now(timezone.utc).isoformat(),
    }
    summary = json.loads(args.summary.read_text())
    merged = {**summary, **extra}
    tmp = args.summary.with_suffix(args.summary.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.summary)
    print(
        json.dumps(
            {
                "event": "official_caliber",
                "rows": len(per),
                "anchor_score_mean": summary.get("score_mean"),
                "score_mean_original": extra["score_mean_original"],
                "neg_hit": f"{neg_hit}/{neg_total}",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
