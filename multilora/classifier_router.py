from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from multilora.router import BaseRouter, RouterDecision

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+|\d+")


def tokenize_router_text(text: str) -> list[str]:
    """Tokenize one router training/example text into stable lowercase units."""

    return [token.casefold() for token in _TOKEN_RE.findall(text)]


@dataclass
class ClassifierRouterArtifact:
    """Portable JSON artifact for the lightweight Naive Bayes router."""

    general_adapter: str
    labels: list[str]
    trained_samples: int
    doc_counts: dict[str, int]
    total_tokens: dict[str, int]
    token_counts: dict[str, dict[str, int]]
    vocabulary: list[str] = field(default_factory=list)


class ClassifierRouter(BaseRouter):
    """Tiny multinomial Naive Bayes router stored as JSON.

    This keeps Level 3 training lightweight enough to run inside the existing
    scheduler CPU pool with no extra heavyweight dependencies.
    """

    def __init__(
        self,
        artifact: ClassifierRouterArtifact,
        *,
        confidence_threshold: float,
    ) -> None:
        self.general_adapter = artifact.general_adapter
        self.labels = list(artifact.labels)
        self.trained_samples = artifact.trained_samples
        self.doc_counts = dict(artifact.doc_counts)
        self.total_tokens = dict(artifact.total_tokens)
        self.token_counts = {
            label: dict(counts)
            for label, counts in artifact.token_counts.items()
        }
        self.vocabulary = set(artifact.vocabulary)
        self.vocab_size = max(1, len(self.vocabulary))
        self.confidence_threshold = confidence_threshold

    def _score_label(self, tokens: list[str], label: str) -> float:
        """Return one log-probability score for a candidate label."""

        total_docs = max(1, sum(self.doc_counts.values()))
        label_docs = self.doc_counts.get(label, 0)
        log_prob = math.log((label_docs + 1) / (total_docs + len(self.labels)))

        label_counts = self.token_counts.get(label, {})
        total_tokens = self.total_tokens.get(label, 0)
        denom = total_tokens + self.vocab_size
        for token in tokens:
            token_count = label_counts.get(token, 0)
            log_prob += math.log((token_count + 1) / denom)
        return log_prob

    @staticmethod
    def _softmax_confidence(scores: dict[str, float], winner: str) -> float:
        """Compute a numerically stable confidence for the winning label."""

        max_score = max(scores.values())
        total = 0.0
        winner_exp = 0.0
        for label, score in scores.items():
            value = math.exp(score - max_score)
            total += value
            if label == winner:
                winner_exp = value
        if total <= 0.0:
            return 0.0
        return winner_exp / total

    def route(self, text: str) -> RouterDecision:
        """Route one text request to the best current type label."""

        if not self.labels:
            return RouterDecision(
                adapter_name=self.general_adapter,
                confidence=0.0,
                rule_name="classifier_empty",
                matched_terms=(),
                fallback_used=True,
            )

        tokens = tokenize_router_text(text)
        scores = {label: self._score_label(tokens, label) for label in self.labels}
        winner = max(scores.items(), key=lambda item: item[1])[0]
        confidence = self._softmax_confidence(scores, winner)
        if confidence < self.confidence_threshold:
            return RouterDecision(
                adapter_name=self.general_adapter,
                confidence=confidence,
                rule_name="classifier_fallback_general",
                matched_terms=tuple(tokens[:8]),
                fallback_used=True,
            )
        return RouterDecision(
            adapter_name=winner,
            confidence=confidence,
            rule_name="classifier",
            matched_terms=tuple(tokens[:8]),
            fallback_used=False,
        )


def train_classifier_router(
    *,
    general_adapter: str,
    labeled_texts: list[tuple[str, str]],
    confidence_threshold: float,
) -> ClassifierRouter | None:
    """Fit one lightweight classifier from labeled text examples."""

    labels = sorted({label for _, label in labeled_texts if label})
    if not labels:
        return None

    doc_counts = {label: 0 for label in labels}
    total_tokens = {label: 0 for label in labels}
    token_counts = {label: {} for label in labels}
    vocabulary: set[str] = set()

    for text, label in labeled_texts:
        if label not in doc_counts:
            continue
        tokens = tokenize_router_text(text)
        doc_counts[label] += 1
        total_tokens[label] += len(tokens)
        label_counts = token_counts[label]
        for token in tokens:
            vocabulary.add(token)
            label_counts[token] = label_counts.get(token, 0) + 1

    artifact = ClassifierRouterArtifact(
        general_adapter=general_adapter,
        labels=labels,
        trained_samples=len(labeled_texts),
        doc_counts=doc_counts,
        total_tokens=total_tokens,
        token_counts=token_counts,
        vocabulary=sorted(vocabulary),
    )
    return ClassifierRouter(artifact, confidence_threshold=confidence_threshold)


def save_classifier_router(router: ClassifierRouter, path: str | Path) -> None:
    """Persist the fitted classifier router to a JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = ClassifierRouterArtifact(
        general_adapter=router.general_adapter,
        labels=router.labels,
        trained_samples=router.trained_samples,
        doc_counts=router.doc_counts,
        total_tokens=router.total_tokens,
        token_counts=router.token_counts,
        vocabulary=sorted(router.vocabulary),
    )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifact), handle, ensure_ascii=False, indent=2, sort_keys=True)


def load_classifier_router(path: str | Path, *, confidence_threshold: float) -> ClassifierRouter:
    """Load one persisted classifier router artifact."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    artifact = ClassifierRouterArtifact(**payload)
    return ClassifierRouter(artifact, confidence_threshold=confidence_threshold)
