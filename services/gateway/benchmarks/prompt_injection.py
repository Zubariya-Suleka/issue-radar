"""Benchmark prompt-injection detectors and save the one the gateway uses (D-017).

    uv sync --all-packages --extra local --extra safety --group bench
    uv run python services/gateway/benchmarks/prompt_injection.py [--with-deberta]

Questions and datasets:
1. Does it catch injections?  deepset/prompt-injections test split (116 rows, EN + DE,
   Apache-2.0). Small: with 60 positives, recall has a 95% interval of roughly +-10 points.
2. Does it catch an injection hidden at the end of a real issue?  Each test injection is
   appended to a held-out pandas issue. This is the threat issue-radar faces (D-003).
3. Does it leave real issue text alone?  Recent issues fetched from the GitHub API into
   data/ (gitignored, D-006), assumed benign. pandas issues are split in half: one half may
   be used for training, the other is held out. scikit-learn issues are never trained on,
   so they show whether the model learned "pandas style" rather than "not an injection".

No choice is made on test data: the TF-IDF regularisation is cross-validated on train.
The saved model is exactly the one measured here (trained on deepset train + half the
pandas issues), so the numbers describe what the gateway runs.
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from gateway.safety.detectors import (
    Detector,
    HeuristicDetector,
    TfidfDetector,
    TransformerDetector,
    paragraph_score,
)

DATASET = "deepset/prompt-injections"
SPLITS = {
    "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "test": "data/test-00000-of-00001-701d16158af87368.parquet",
}
DEBERTA = "protectai/deberta-v3-base-prompt-injection-v2"
OUT_DIR = Path("data/benchmarks")
MODEL_PATH = Path("models/prompt_injection_tfidf.joblib")
THRESHOLD = 0.5

# Crude language split, good enough to show whether a detector is English-only.
_GERMAN = re.compile(r"[äöüß]|\b(und|ist|nicht|der|die|das|ich|du|sie|wie|was|bitte)\b", re.I)


def load_split(split: str) -> list[dict]:
    path = hf_hub_download(DATASET, SPLITS[split], repo_type="dataset")
    return pq.read_table(path).to_pylist()


def load_issues(repo: str, n_pages: int = 4) -> list[str]:
    """Recent issues (not PRs) as 'title + body'. Cached after the first run."""
    cache = OUT_DIR / f"{repo.replace('/', '__')}_issues.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    texts: list[str] = []
    for page in range(1, n_pages + 1):
        # The search API filters PRs out server-side; the /issues endpoint mixes them in.
        url = (
            f"https://api.github.com/search/issues?q=repo:{repo}+is:issue"
            f"&sort=created&order=desc&per_page=100&page={page}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.load(resp)["items"]
        texts += [f"{i['title']}\n\n{i['body'] or ''}" for i in items]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(texts), encoding="utf-8")
    return texts


def prf(pred: list[bool], gold: list[int]) -> dict[str, float]:
    tp = sum(p and g for p, g in zip(pred, gold, strict=True))
    fp = sum(p and not g for p, g in zip(pred, gold, strict=True))
    fn = sum(not p and g for p, g in zip(pred, gold, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def flag_rate(score: Callable[[str], float], texts: list[str]) -> tuple[float, float, list[str]]:
    """Share of texts flagged, median ms per text, and a few flagged examples."""
    timings, flagged = [], []
    for text in texts:
        start = time.perf_counter()
        if score(text) >= THRESHOLD:
            flagged.append(text)
        timings.append((time.perf_counter() - start) * 1000)
    return len(flagged) / len(texts), statistics.median(timings), flagged[:5]


def evaluate(detector: Detector, per_paragraph: bool, data: dict) -> dict:
    def score(text: str) -> float:
        return paragraph_score(detector, text) if per_paragraph else detector.score(text)

    test = data["test"]
    pred = [score(r["text"]) >= THRESHOLD for r in test]
    gold = [r["label"] for r in test]
    german = [bool(_GERMAN.search(r["text"])) for r in test]

    def subset(keep_german: bool) -> dict[str, float]:
        rows = [(p, g) for p, g, de in zip(pred, gold, german, strict=True) if de == keep_german]
        return prf([p for p, _ in rows], [g for _, g in rows])

    hidden_recall, _, _ = flag_rate(score, data["hidden"])
    pandas_fp, ms_per_issue, pandas_examples = flag_rate(score, data["pandas"])
    sklearn_fp, _, sklearn_examples = flag_rate(score, data["sklearn"])
    return {
        "detector": detector.name + (" /para" if per_paragraph else ""),
        "test": prf(pred, gold),
        "test_en": subset(False),
        "test_de": subset(True),
        "hidden_recall": hidden_recall,
        "fp_pandas": pandas_fp,
        "fp_sklearn": sklearn_fp,
        "ms_per_issue": ms_per_issue,
        "flagged_examples": [t[:200] for t in pandas_examples + sklearn_examples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-deberta", action="store_true", help="also run DeBERTa (~15 min on CPU)"
    )
    args = parser.parse_args()
    # Issue text contains characters a Windows console code page can't encode.
    sys.stdout.reconfigure(encoding="utf-8")

    train, test = load_split("train"), load_split("test")
    pandas = load_issues("pandas-dev/pandas")
    sklearn = load_issues("scikit-learn/scikit-learn")
    pandas_train, pandas_held_out = pandas[::2], pandas[1::2]
    rng = random.Random(0)
    hidden = [f"{rng.choice(pandas_held_out)}\n\n{r['text']}" for r in test if r["label"] == 1]
    data = {"test": test, "hidden": hidden, "pandas": pandas_held_out, "sklearn": sklearn}
    print(
        f"deepset train {len(train)} / test {len(test)}; pandas issues {len(pandas_train)} "
        f"train / {len(pandas_held_out)} held out; scikit-learn issues {len(sklearn)} (unseen)\n"
    )

    deepset_texts, deepset_labels = [r["text"] for r in train], [r["label"] for r in train]
    tfidf_deepset = TfidfDetector.train(deepset_texts, deepset_labels)
    tfidf_deepset.name = "tfidf (deepset only)"
    tfidf = TfidfDetector.train(
        deepset_texts + pandas_train, deepset_labels + [0] * len(pandas_train)
    )
    tfidf.name = "tfidf (+ pandas issues as benign)"

    detectors: list[Detector] = [HeuristicDetector(), tfidf_deepset, tfidf]
    if args.with_deberta:
        detectors.append(TransformerDetector(DEBERTA))
    results = [evaluate(d, para, data) for d in detectors for para in (False, True)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prompt_injection_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tfidf.save(MODEL_PATH)

    print(
        "| detector | P | R | F1 | R (EN) | R (DE) | R hidden in issue "
        "| FP pandas | FP scikit-learn | ms / issue |"
    )
    print("|---" * 10 + "|")
    for r in results:
        t = r["test"]
        print(
            f"| {r['detector']} | {t['precision']:.2f} | {t['recall']:.2f} | {t['f1']:.2f} "
            f"| {r['test_en']['recall']:.2f} | {r['test_de']['recall']:.2f} "
            f"| {r['hidden_recall']:.2f} | {r['fp_pandas']:.1%} | {r['fp_sklearn']:.1%} "
            f"| {r['ms_per_issue']:.1f} |"
        )
    print(f"\nSaved the gateway model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
