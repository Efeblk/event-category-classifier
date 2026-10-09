"""Probe real, unlabeled news excerpts and input perturbations without fitting.

The pinned archive's official dev articles were never used to train these models.
Their template has no technique gold, so this reports behavior, not accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile
from typing import Any

from method_comparison import METHODS, compare_request, validate_artifacts
from prepare_data import SHA256, URL, digest, normalize_whitespace


ARTICLE_PATTERN = re.compile(r"datasets/dev-articles/article([0-9]+)\.txt")
TEMPLATE = "datasets/dev-task-TC-template.out"
METHOD_SOURCE = "https://aclanthology.org/2020.acl-main.442/"
SOURCE_PAGE = "https://zenodo.org/records/3952415"


def build_inputs(archive: str | Path, count: int = 25, seed: int = 42) -> list[dict[str, Any]]:
    """Read verified regular dev members only, selecting one span per article."""
    if type(count) is not int or count < 1:
        raise ValueError("Choose at least one human news excerpt.")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer.")
    if digest(archive) != SHA256:
        raise ValueError("Archive checksum mismatch; no robustness source was read.")
    with tarfile.open(archive, "r:gz") as source:
        members = [m for m in source.getmembers() if ARTICLE_PATTERN.fullmatch(m.name) or m.name == TEMPLATE]
        if len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members):
            raise ValueError("Robustness sources must be unique regular files.")
        by_name = {m.name: m for m in members}
        if TEMPLATE not in by_name:
            raise ValueError("The official unlabeled dev template is missing.")
        annotations: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for line in source.extractfile(by_name[TEMPLATE]).read().decode("utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 4 or not fields[0].isdigit() or fields[1] != "?":
                raise ValueError("Expected an unlabeled four-column official dev template.")
            try:
                start, end = int(fields[2]), int(fields[3])
            except ValueError as error:
                raise ValueError("Official dev offsets must be integers.") from error
            if not 0 <= start < end:
                raise ValueError("Official dev offsets must be increasing and nonnegative.")
            annotations[fields[0]].append((start, end))
        article_ids = sorted(annotations)
        if count > len(article_ids):
            raise ValueError(f"Only {len(article_ids)} annotated dev articles are available.")
        rng = random.Random(seed)
        rng.shuffle(article_ids)
        rows = []
        for article_id in article_ids[:count]:
            member_name = f"datasets/dev-articles/article{article_id}.txt"
            if member_name not in by_name:
                raise ValueError("A dev template article is missing from the archive.")
            raw = source.extractfile(by_name[member_name]).read()
            article = raw.decode("utf-8")
            spans = annotations[article_id]
            if any(end > len(article) for _, end in spans):
                raise ValueError("Official dev offsets exceed the original article.")
            start, end = rng.choice(spans)
            text = article[start:end]
            if not text.strip():
                raise ValueError("The dev template contains an empty excerpt.")
            rows.append({
                "id": f"official_dev:{article_id}:{start}:{end}", "article_id": article_id,
                "text": text, "context": normalize_whitespace(article[max(0, start - 1000):end + 1000]),
                "start": start, "end": end, "gold": None,
                "origin": "human_written_news", "source_partition": "official_dev_unlabeled",
                "source_url": SOURCE_PAGE, "source_member": member_name,
                "source_sha256": hashlib.sha256(raw).hexdigest(), "license": "CC-BY-4.0",
            })
    return sorted(rows, key=lambda row: row["id"])


def typo(text: str) -> str:
    """Swap the first two letters of one longer word as a typing diagnostic."""
    return re.sub(r"\b([A-Za-z])([A-Za-z])([A-Za-z]{2,})\b",
                  lambda match: match[2] + match[1] + match[3], text, count=1)


def variants(row: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Changes are diagnostics, not new human-authored or gold-labeled examples."""
    text, context = row["text"], row["context"]
    return {
        "original": (text, context),
        "whitespace": (" \n" + re.sub(r"\s+", "\t ", text) + "\n ", context),
        "uppercase": (text.upper(), context),
        "curly_apostrophe": (text.translate(str.maketrans({"'": "\u2019", "\u2019": "'"})), context),
        "one_typo": (typo(text), context),
        "emoji_suffix": (text + " \U0001f642", context),
        "no_context": (text, ""),
        "unrelated_context": (text, "The train leaves at six. The weather forecast says rain tomorrow."),
    }


def _predictions(artifacts: dict, text: str, context: str) -> dict[str, dict[str, Any]]:
    result = compare_request(artifacts, text, context=context)
    return {row["method"]: {key: row[key] for key in ("label", "reason") if key in row}
            for row in result["results"] if row["method"] in METHODS}


def probe(artifacts: dict, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Record label changes and a few synthetic guards, never technique accuracy."""
    validate_artifacts(artifacts)
    if not rows or any(row.get("gold") is not None for row in rows):
        raise ValueError("This unlabeled probe needs nonempty inputs with no gold labels.")
    changes = {name: Counter() for name in variants(rows[0]) if name != "original"}
    modified = Counter()
    comparisons = []
    for row in rows:
        inputs = variants(row)
        outputs = {name: _predictions(artifacts, text, context)
                   for name, (text, context) in inputs.items()}
        original = outputs["original"]
        for name, predictions in outputs.items():
            if name != "original":
                modified[name] += inputs[name] != inputs["original"]
                for method in METHODS:
                    changes[name][method] += predictions[method]["label"] != original[method]["label"]
        comparisons.append({"id": row["id"], "predictions": outputs})
    controls = []
    known_context = "The government must act. Corrupt leaders betrayed the people."
    for name, text, expected in (
        ("unknown_excerpt_known_context", "qxzvqxzv", "unclear"),
        ("emoji_excerpt_known_context", "\U0001f984\U0001fa90\U0001f6f8", "unclear"),
        ("neutral_conversation", "Hello, can you help me with this?", None),
        ("non_english_latin", "Bu teklif herkes i\u00e7in uygun mu?", None),
        ("digits", "2026", None),
        ("ellipsis", "...", None),
    ):
        predictions = _predictions(artifacts, text, known_context if expected else "")
        controls.append({"id": name, "origin": "synthetic_control", "text": text,
                         "context": known_context if expected else "", "expected_label": expected,
                         "predictions": predictions,
                         "passed": all(p["label"] == expected for p in predictions.values()) if expected else None})
    first = next(iter(artifacts.values()))
    return {
        "schema_version": 1, "evaluation_scope": "unlabeled_behavioral_diagnostics",
        "human_source_count": len(rows), "human_article_count": len({row["article_id"] for row in rows}),
        "diagnostic_case_count": len(rows) * len(variants(rows[0])),
        "transformed_case_count": len(rows) * (len(variants(rows[0])) - 1),
        "synthetic_control_count": len(controls),
        "paid_calls": 0, "accuracy": None, "fit_performed": False,
        "training_data_sha256": first["dataset_sha256"], "training_code_sha256": first["code_sha256"],
        "training_seed": first["seed"], "model_evaluation_scope": first["provenance"]["evaluation_scope"],
        "context_margin": first["provenance"]["context_margin"],
        "method_models": {name: artifact["model_name"] for name, artifact in artifacts.items()},
        "methodology_source": METHOD_SOURCE,
        "note": "Human sources have no technique gold. Perturbations and controls are generated. "
                "Label changes diagnose sensitivity, not errors or generalization accuracy. "
                "These examined inputs are not a fresh final test.",
        "sources": rows,
        "label_changes": {name: {method: {"count": counts[method], "pairs": modified[name],
                                          "unchanged_input_pairs": len(rows) - modified[name],
                                          "fraction": counts[method] / modified[name] if modified[name] else None}
                                 for method in METHODS}
                          for name, counts in changes.items()},
        "comparisons": comparisons, "controls": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/source/datasets-v2.tgz"))
    parser.add_argument("--models", type=Path, default=Path("artifacts/method_models.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/robustness.json"))
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        import joblib

        rows = build_inputs(args.archive, args.count, args.seed)
        report = probe(joblib.load(args.models), rows)  # Trusted local train.py artifact only.
        report.update(source_archive_url=URL, source_archive_sha256=SHA256, seed=args.seed)
        report["script_sha256"] = {name: digest(Path(__file__).resolve().parent / name)
                                   for name in ("stress_test.py", "method_comparison.py", "classifier.py")}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"report": str(args.report), "human_sources": len(rows),
                          "diagnostic_cases": report["diagnostic_case_count"],
                          "transformed_cases": report["transformed_case_count"], "accuracy": None,
                          "synthetic_guards_passed": all(row["passed"] is not False for row in report["controls"])}))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
