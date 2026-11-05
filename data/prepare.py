import argparse
import json
from pathlib import Path


def prepare(source, domain, split):
    from math import isfinite

    rows = []
    ids = set()
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        pid, query = str(item["id"]), item["query"]
        if not pid or pid in ids or not isinstance(query, str) or not query.strip():
            raise ValueError("each item needs a unique ID and nonempty query")
        ids.add(pid)
        rubrics = []
        for index, criterion in enumerate(item["rubrics"]):
            weight = criterion["weight"]
            text = criterion["description"]
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not isfinite(weight)
                or weight == 0
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(
                    "rubrics require nonzero finite weights and nonempty descriptions"
                )
            rubrics.append(
                {
                    "rubric_key": f"{pid}:r{index}",
                    "title": criterion.get("title") or f"Criterion {index + 1}",
                    "description": text,
                    "weight": weight,
                }
            )
        if not any(r["weight"] > 0 for r in rubrics):
            raise ValueError("at least one positive-weight rubric is required")
        messages = [{"role": "user", "content": query}]
        rows.append(
            {
                "data_source": {
                    "medicine": "rar_medicine_static_rubric",
                    "science": "rar_science_static_rubric",
                }[domain],
                "prompt": messages,
                "ability": domain,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "prompt_id": pid,
                        "query": query,
                        "rubrics": rubrics,
                    },
                },
                "extra_info": {
                    "student_messages": messages,
                    "split": split,
                    "reference_answer": item.get("reference_answer", ""),
                },
            }
        )
    if not rows:
        raise ValueError("source must not be empty")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=["medicine", "science"], required=True)
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], required=True
    )
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = prepare(args.input, args.domain, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
