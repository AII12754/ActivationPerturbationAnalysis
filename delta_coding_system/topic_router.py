from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "topic:c_programming": ("c", "c语言", "gcc", "clang", "pointer", "malloc", "segfault", "header", "struct"),
    "topic:cpp": ("c++", "cpp", "template", "stl", "vector", "std", "namespace"),
    "topic:python": ("python", "pytorch", "numpy", "pandas", "script", "函数", "脚本"),
    "topic:animals": ("animal", "animals", "cat", "dog", "bird", "tiger", "lion", "elephant", "猫", "狗"),
    "topic:biology": ("biology", "cell", "gene", "protein", "species", "evolution", "生态"),
    "topic:math": ("math", "algebra", "geometry", "theorem", "proof", "证明", "矩阵", "积分", "导数"),
    "topic:physics": ("physics", "quantum", "force", "energy", "particle", "相对论"),
    "topic:chemistry": ("chemistry", "molecule", "chemical", "reaction", "polymer", "化学"),
    "topic:summarization": ("summary", "summarize", "总结", "摘要", "tl;dr"),
    "topic:retrieval": ("retrieve", "retrieval", "evidence", "passage", "search", "检索"),
    "topic:qa": ("question", "answer", "qa", "问答", "hotpotqa", "triviaqa"),
}


def _normalize_domain(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_:+-]+", "_", value.strip().lower())
    return normalized.strip("_") or "misc"


def _normalize_vector(vector: Dict[str, float]) -> Dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 1e-8:
        return {}
    return {key: value / norm for key, value in vector.items()}


@dataclass
class RoutingDecision:
    active_domains: List[str]
    domain_weights: Dict[str, float]
    write_domains: List[str]
    task_domain: Optional[str]
    topic_scores: List[Tuple[str, float]] = field(default_factory=list)
    merge_domains: List[str] = field(default_factory=list)


class LightweightTopicRouter:
    def __init__(
        self,
        topic_keywords: Optional[Dict[str, Sequence[str]]] = None,
        max_active_domains: int = 4,
        max_write_domains: int = 3,
        min_topic_score: float = 0.12,
        merge_threshold: float = 8.0,
        split_threshold: float = 2.5,
        ema_decay: float = 0.92,
        prototype_momentum: float = 0.25,
    ):
        self.max_active_domains = max(1, max_active_domains)
        self.max_write_domains = max(1, max_write_domains)
        self.min_topic_score = min_topic_score
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.ema_decay = ema_decay
        self.prototype_momentum = prototype_momentum

        raw_keywords = topic_keywords or DEFAULT_TOPIC_KEYWORDS
        self.topic_keywords = {
            _normalize_domain(domain): tuple(keyword.lower() for keyword in keywords)
            for domain, keywords in raw_keywords.items()
        }
        self.domain_prototypes: Dict[str, Dict[str, float]] = {}
        self.domain_stats: Dict[str, Dict[str, float]] = {}
        self.pair_coactivation: Dict[Tuple[str, str], float] = {}
        self.merge_groups: Dict[str, Tuple[str, str]] = {}

        for domain, keywords in self.topic_keywords.items():
            seed = {f"kw:{keyword}": 1.0 for keyword in keywords}
            token_counts = self._extract_features(" ".join(keywords))
            merged = dict(seed)
            for key, value in token_counts.items():
                merged[key] = merged.get(key, 0.0) + value
            self.domain_prototypes[domain] = _normalize_vector(merged)
            self.domain_stats[domain] = {"requests": 0.0, "hits": 0.0}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_+#\-.]+|[\u4e00-\u9fff]{1,4}", text.lower())

    def _extract_features(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        features: Dict[str, float] = {}
        for token in tokens:
            features[f"tok:{token}"] = features.get(f"tok:{token}", 0.0) + 1.0
            if len(token) >= 3:
                for idx in range(len(token) - 2):
                    trigram = token[idx : idx + 3]
                    features[f"tri:{trigram}"] = features.get(f"tri:{trigram}", 0.0) + 0.35
        return _normalize_vector(features)

    def _cosine_similarity(self, left: Dict[str, float], right: Dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(key, 0.0) for key, value in left.items())

    def _compose_merge_domain(self, domain_a: str, domain_b: str) -> str:
        left, right = sorted((_normalize_domain(domain_a), _normalize_domain(domain_b)))
        return f"merge:{left}+{right}"

    def _register_merge_domain(self, domain_a: str, domain_b: str) -> str:
        merge_domain = self._compose_merge_domain(domain_a, domain_b)
        if merge_domain in self.merge_groups:
            return merge_domain
        left, right = sorted((_normalize_domain(domain_a), _normalize_domain(domain_b)))
        self.merge_groups[merge_domain] = (left, right)
        proto_left = self.domain_prototypes.get(left, {})
        proto_right = self.domain_prototypes.get(right, {})
        merged: Dict[str, float] = {}
        for key, value in proto_left.items():
            merged[key] = merged.get(key, 0.0) + 0.5 * value
        for key, value in proto_right.items():
            merged[key] = merged.get(key, 0.0) + 0.5 * value
        self.domain_prototypes[merge_domain] = _normalize_vector(merged)
        self.domain_stats.setdefault(merge_domain, {"requests": 0.0, "hits": 0.0})
        return merge_domain

    def _drop_merge_domain(self, merge_domain: str) -> None:
        self.merge_groups.pop(merge_domain, None)
        self.domain_prototypes.pop(merge_domain, None)
        self.domain_stats.pop(merge_domain, None)

    def _update_merge_groups(self) -> None:
        for pair, score in list(self.pair_coactivation.items()):
            domain_a, domain_b = pair
            merge_domain = self._compose_merge_domain(domain_a, domain_b)
            if score >= self.merge_threshold:
                self._register_merge_domain(domain_a, domain_b)
            elif score < self.split_threshold and merge_domain in self.merge_groups:
                self._drop_merge_domain(merge_domain)

    def route(
        self,
        text: str,
        task_name: Optional[str] = None,
        manual_domains: Optional[Sequence[str]] = None,
    ) -> RoutingDecision:
        features = self._extract_features(text)
        ordered_domains: List[str] = []
        seen = set()
        domain_weights: Dict[str, float] = {}
        topic_scores: List[Tuple[str, float]] = []

        def add_domain(domain: str, weight: float) -> None:
            normalized = _normalize_domain(domain)
            domain_weights[normalized] = max(domain_weights.get(normalized, 0.0), weight)
            if normalized not in seen:
                seen.add(normalized)
                ordered_domains.append(normalized)

        task_domain = None
        for domain in manual_domains or []:
            add_domain(domain, 1.35)

        if task_name:
            task_domain = _normalize_domain(f"task:{task_name}")
            add_domain(task_domain, 1.5)
            self.domain_stats.setdefault(task_domain, {"requests": 0.0, "hits": 0.0})
            self.domain_prototypes.setdefault(task_domain, {})

        scored_domains: List[Tuple[str, float]] = []
        for domain, prototype in self.domain_prototypes.items():
            if domain.startswith("task:"):
                continue
            score = self._cosine_similarity(features, prototype)
            prior = self.domain_stats.get(domain, {}).get("hits", 0.0)
            if prior > 0:
                score += min(0.25, math.log1p(prior) * 0.03)
            if score >= self.min_topic_score:
                scored_domains.append((domain, score))

        scored_domains.sort(key=lambda item: (-item[1], item[0]))
        topic_scores = scored_domains[: self.max_active_domains * 2]
        for domain, score in topic_scores:
            add_domain(domain, 0.55 + score)
            if len(ordered_domains) >= self.max_active_domains:
                break

        active_domains = ordered_domains[: self.max_active_domains]
        merge_domains = [domain for domain in active_domains if domain.startswith("merge:")]

        write_candidates = sorted(
            ((domain, domain_weights.get(domain, 0.0)) for domain in active_domains),
            key=lambda item: (-item[1], item[0]),
        )
        write_domains = [domain for domain, _weight in write_candidates[: self.max_write_domains]]
        if task_domain and task_domain not in write_domains:
            write_domains = [task_domain] + write_domains[: self.max_write_domains - 1]

        return RoutingDecision(
            active_domains=active_domains,
            domain_weights={domain: domain_weights[domain] for domain in active_domains},
            write_domains=write_domains,
            task_domain=task_domain,
            topic_scores=topic_scores,
            merge_domains=merge_domains,
        )

    def observe(
        self,
        text: str,
        routing: RoutingDecision,
        domain_hits: Dict[str, int],
    ) -> None:
        features = self._extract_features(text)
        active_domains = [domain for domain in routing.active_domains if not domain.startswith("task:")]

        for pair, current in list(self.pair_coactivation.items()):
            self.pair_coactivation[pair] = current * self.ema_decay
        for index, left in enumerate(active_domains):
            for right in active_domains[index + 1 :]:
                key = tuple(sorted((left, right)))
                self.pair_coactivation[key] = self.pair_coactivation.get(key, 0.0) * self.ema_decay + 1.0

        for domain in routing.active_domains:
            stats = self.domain_stats.setdefault(domain, {"requests": 0.0, "hits": 0.0})
            stats["requests"] += 1.0
            stats["hits"] += float(domain_hits.get(domain, 0))

        for domain in routing.write_domains:
            weight = routing.domain_weights.get(domain, 0.0)
            if weight <= 0:
                continue
            old_proto = self.domain_prototypes.get(domain, {})
            blended: Dict[str, float] = {key: value * (1.0 - self.prototype_momentum) for key, value in old_proto.items()}
            for key, value in features.items():
                blended[key] = blended.get(key, 0.0) + self.prototype_momentum * weight * value
            self.domain_prototypes[domain] = _normalize_vector(blended)

        self._update_merge_groups()
