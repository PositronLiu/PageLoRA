"""Lightweight rule-based logical-adapter router.

Level 2 only needs an inexpensive and explainable routing baseline:

- no model calls
- no scheduler interaction
- no top-K / RL / cost-aware logic

The router consumes raw task text and returns a single primary logical adapter
decision plus a few diagnostics for logging and offline analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


@dataclass(frozen=True)
class RuleRouteSpec:
    """One explicit routing rule for a target logical adapter."""

    adapter_name: str
    rule_name: str
    keywords: tuple[str, ...]
    regexes: tuple[str, ...] = ()
    score: float = 1.0


@dataclass(frozen=True)
class RouterDecision:
    """Result of one routing decision."""

    adapter_name: str
    confidence: float
    rule_name: str
    matched_terms: tuple[str, ...]
    fallback_used: bool


class BaseRouter:
    """Minimal router interface."""

    def route(self, text: str) -> RouterDecision:
        raise NotImplementedError


class RuleRouter(BaseRouter):
    """Keyword/regex-based logical-adapter router.

    This router is intentionally small and deterministic. It is used as the
    Level 2 baseline before introducing learned or cost-aware routing.
    """

    def __init__(
        self,
        *,
        general_adapter: str,
        rules: tuple[RuleRouteSpec, ...],
        fallback_threshold: float = 0.5,
    ) -> None:
        self.general_adapter = general_adapter
        self.fallback_threshold = fallback_threshold
        self._compiled_rules: tuple[tuple[RuleRouteSpec, tuple[str, ...], tuple[Pattern[str], ...]], ...] = tuple(
            (
                rule,
                tuple(keyword.casefold() for keyword in rule.keywords),
                tuple(re.compile(pattern, re.IGNORECASE) for pattern in rule.regexes),
            )
            for rule in rules
        )

    def _score_rule(
        self,
        text: str,
        rule: RuleRouteSpec,
        keywords: tuple[str, ...],
        regexes: tuple[Pattern[str], ...],
    ) -> tuple[float, tuple[str, ...]]:
        """Score one rule and return the normalized confidence plus matches."""

        matched_terms: list[str] = []
        keyword_hits = 0
        regex_hits = 0

        for keyword in keywords:
            if keyword and keyword in text:
                keyword_hits += 1
                matched_terms.append(keyword)

        for pattern in regexes:
            match = pattern.search(text)
            if match is not None:
                regex_hits += 1
                matched_terms.append(pattern.pattern)

        if keyword_hits == 0 and regex_hits == 0:
            return 0.0, ()

        confidence = min(1.0, rule.score * (0.35 * keyword_hits + 0.2 * regex_hits))
        return confidence, tuple(matched_terms)

    def route(self, text: str) -> RouterDecision:
        """Choose one logical adapter for the current input text."""

        text_lower = text.casefold()
        best_adapter = self.general_adapter
        best_rule = "fallback_general"
        best_confidence = 0.0
        best_terms: tuple[str, ...] = ()

        for rule, keywords, regexes in self._compiled_rules:
            confidence, matched_terms = self._score_rule(text_lower, rule, keywords, regexes)
            if confidence > best_confidence:
                best_adapter = rule.adapter_name
                best_rule = rule.rule_name
                best_confidence = confidence
                best_terms = matched_terms

        if best_confidence < self.fallback_threshold:
            return RouterDecision(
                adapter_name=self.general_adapter,
                confidence=best_confidence,
                rule_name="fallback_general",
                matched_terms=best_terms,
                fallback_used=True,
            )

        return RouterDecision(
            adapter_name=best_adapter,
            confidence=best_confidence,
            rule_name=best_rule,
            matched_terms=best_terms,
            fallback_used=False,
        )


def build_default_rule_router(
    *,
    logical_adapters: tuple[str, ...],
    general_adapter: str,
    fallback_threshold: float,
) -> RuleRouter:
    """Build the default Level 2 rule router from registered adapters only."""

    available = set(logical_adapters)
    rules: list[RuleRouteSpec] = []

    if "coding" in available:
        rules.append(
            RuleRouteSpec(
                adapter_name="coding",
                rule_name="coding_keyword",
                keywords=(
                    "traceback",
                    "exception",
                    "error",
                    "bug",
                    "stack trace",
                    "python",
                    "javascript",
                    "代码",
                    "函数",
                    "类",
                    "编译",
                    "报错",
                    "算法",
                ),
                regexes=(r"\bdef\b", r"\bclass\b", r"```[\w-]*"),
            )
        )

    if "email" in available:
        rules.append(
            RuleRouteSpec(
                adapter_name="email",
                rule_name="email_keyword",
                keywords=(
                    "email",
                    "邮件",
                    "dear",
                    "subject",
                    "收件人",
                    "发件人",
                    "润色邮件",
                ),
            )
        )

    if "terminal" in available:
        rules.append(
            RuleRouteSpec(
                adapter_name="terminal",
                rule_name="terminal_keyword",
                keywords=(
                    "bash",
                    "shell",
                    "linux",
                    "docker",
                    "kubectl",
                    "apt",
                    "conda",
                    "terminal",
                    "命令行",
                    "脚本",
                ),
            )
        )

    return RuleRouter(
        general_adapter=general_adapter,
        rules=tuple(rules),
        fallback_threshold=fallback_threshold,
    )


__all__ = [
    "BaseRouter",
    "RouterDecision",
    "RuleRouteSpec",
    "RuleRouter",
    "build_default_rule_router",
]
