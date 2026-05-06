from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.utils import WARNING
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
    artifact_version: int = 2
    updated_at: float = 0.0


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

    @classmethod
    def empty(
        cls,
        *,
        general_adapter: str,
        confidence_threshold: float,
    ) -> "ClassifierRouter":
        """Create an empty incremental classifier."""

        artifact = ClassifierRouterArtifact(
            general_adapter=general_adapter,
            labels=[],
            trained_samples=0,
            doc_counts={},
            total_tokens={},
            token_counts={},
            vocabulary=[],
            updated_at=time.time(),
        )
        return cls(artifact, confidence_threshold=confidence_threshold)

    @classmethod
    def from_artifact(
        cls,
        artifact: ClassifierRouterArtifact,
        *,
        confidence_threshold: float,
    ) -> "ClassifierRouter":
        """Restore a classifier from a persisted artifact."""

        return cls(artifact, confidence_threshold=confidence_threshold)

    def to_artifact(self) -> ClassifierRouterArtifact:
        """Export the current classifier state for persistence."""

        return ClassifierRouterArtifact(
            general_adapter=self.general_adapter,
            labels=list(self.labels),
            trained_samples=self.trained_samples,
            doc_counts=dict(self.doc_counts),
            total_tokens=dict(self.total_tokens),
            token_counts={
                label: dict(counts)
                for label, counts in self.token_counts.items()
            },
            vocabulary=sorted(self.vocabulary),
            updated_at=time.time(),
        )

    def clone(self) -> "ClassifierRouter":
        """Return a detached copy that can be trained off the foreground path."""

        return ClassifierRouter.from_artifact(
            self.to_artifact(),
            confidence_threshold=self.confidence_threshold,
        )

    @classmethod
    def load(cls, path: str | Path, *, confidence_threshold: float) -> "ClassifierRouter":
        """Load one persisted classifier router artifact."""

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("artifact_version", 1)
        payload.setdefault("updated_at", 0.0)
        artifact = ClassifierRouterArtifact(**payload)
        return cls.from_artifact(artifact, confidence_threshold=confidence_threshold)

    def save(self, path: str | Path) -> None:
        """Persist the current classifier router artifact."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.to_artifact()), handle, ensure_ascii=False, indent=2, sort_keys=True)

    def train_one(self, text: str, label: str) -> bool:
        """Incrementally absorb one labeled router example."""

        label = label.strip()
        if not label:
            return False
        tokens = tokenize_router_text(text)
        if label not in self.doc_counts:
            self.labels.append(label)
            self.labels.sort()
            self.doc_counts[label] = 0
            self.total_tokens[label] = 0
            self.token_counts[label] = {}

        self.doc_counts[label] += 1
        self.total_tokens[label] += len(tokens)
        label_counts = self.token_counts[label]
        for token in tokens:
            self.vocabulary.add(token)
            label_counts[token] = label_counts.get(token, 0) + 1
        self.vocab_size = max(1, len(self.vocabulary))
        self.trained_samples += 1
        return True

    def train_batch(self, labeled_texts: list[tuple[str, str]]) -> int:
        """Incrementally absorb a batch of labeled router examples."""

        trained = 0
        for text, label in labeled_texts:
            if self.train_one(text, label):
                trained += 1
        return trained

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

