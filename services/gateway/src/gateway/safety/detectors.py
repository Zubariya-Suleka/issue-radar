"""Prompt-injection detectors (D-017).

A detector maps text to a score in [0, 1]; the caller compares it with a threshold. The
benchmark (services/gateway/benchmarks/prompt_injection.py) runs these same classes, so
its numbers describe exactly what the gateway runs.
"""

import re
from pathlib import Path
from typing import Any, Protocol


class Detector(Protocol):
    name: str

    def score(self, text: str) -> float: ...


# Phrases that redirect the model away from its instructions, in English and German
# (deepset/prompt-injections is roughly half German). Deliberately narrow: GitHub issues
# say "ignore" and "instructions" all the time, so broad keywords would flag normal text.
_PATTERNS = [
    # "ignore / forget / disregard (all) (the) previous / above / prior instructions"
    r"\b(ignore|forget|disregard|overlook)\b.{0,30}\b(previous|above|prior|earlier|all)\b"
    r".{0,20}\b(instructions?|prompts?|tasks?|rules|context|information|everything)\b",
    r"\b(forget|ignore)\s+(everything|all)\b",
    r"\b(new|next)\s+(task|instructions?|assignment)\s*[:!]",
    r"\byou\s+are\s+now\b",
    r"\b(pretend|act)\s+(to\s+be|as\s+if|as)\b",
    r"\b(reveal|print|show|repeat|output)\b.{0,30}\b(system\s+prompt|instructions|prompt\s+above|"
    r"text\s+above|initial\s+prompt)\b",
    # German
    r"\b(vergiss|vergessen\s+sie|ignoriere|ignorieren\s+sie)\b.{0,40}\b(alles|alle|vorherigen|"
    r"bisherigen|vorangegangenen)\b",
    r"\b(neue|nächste)\s+aufgabe\b",
    r"\b(jetzt|nun)\s+bist\s+du\b",
    r"\bdu\s+bist\s+(jetzt|nun)\b",
]
_REGEX = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.DOTALL)


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def paragraph_score(detector: Detector, text: str) -> float:
    """Highest score over the text's paragraphs.

    Scoring a long issue as one text lets its ordinary content outvote a short injected
    paragraph, and a 512-token classifier never sees text past its limit. Scoring each
    paragraph avoids both, at the cost of one call per paragraph.
    """
    paragraphs = [p for p in _PARAGRAPH_BREAK.split(text) if p.strip()] or [text]
    return max(detector.score(p) for p in paragraphs)


class HeuristicDetector:
    """Regex rules. No dependencies, microseconds per call, easy to explain; misses paraphrases."""

    name = "heuristic"

    def score(self, text: str) -> float:
        return 1.0 if _REGEX.search(text) else 0.0


class TfidfDetector:
    """Character n-gram TF-IDF + logistic regression. Sub-millisecond on CPU.

    Character n-grams handle English/German mixing and typos without a per-language
    tokenizer. Needs scikit-learn (the gateway's 'safety' extra). Trained and saved by the
    benchmark; saved with joblib, which is pickle: load only files you trained yourself.
    """

    name = "tfidf"

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    @classmethod
    def train(cls, texts: list[str], labels: list[int]) -> "TfidfDetector":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.pipeline import make_pipeline

        pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
            # Regularisation picked by 5-fold cross-validation on the training data only.
            LogisticRegressionCV(
                Cs=10,
                cv=5,
                scoring="f1",
                max_iter=2000,
                l1_ratios=(0.0,),  # plain L2; explicit to opt in to scikit-learn 1.10 behaviour
                use_legacy_attributes=False,
            ),
        )
        pipeline.fit(texts, labels)
        return cls(pipeline)

    @classmethod
    def load(cls, path: Path) -> "TfidfDetector":
        import joblib

        return cls(joblib.load(path))

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    def score(self, text: str) -> float:
        return float(self._pipeline.predict_proba([text])[0, 1])


class TransformerDetector:
    """A fine-tuned sequence classifier, e.g. protectai/deberta-v3-base-prompt-injection-v2.

    Needs the 'local' extra (torch, transformers). Truncates to the model's 512 tokens, so
    an injection placed after a long preamble can be cut off.
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Path | None = None,
        injection_label: str = "INJECTION",
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.name = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self._model.eval()
        self._label_index = self._model.config.label2id[injection_label]

    def score(self, text: str) -> float:
        inputs = self._tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        return logits.softmax(dim=-1)[0, self._label_index].item()
