from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
import itertools
import json
import re
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from evidencelink.utils.text import normalize_structure_text


@dataclass(frozen=True)
class EvidenceNeed:
    unit_id: str
    subquery: str
    depends_on: Tuple[str, ...] = ()
    expected_answer_type: str = "unknown"
    anchor_mentions: Tuple[str, ...] = ()
    role: str = "support"
    satisfiable_by: str = "unknown"

    def to_trace(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["depends_on"] = list(self.depends_on)
        payload["anchor_mentions"] = list(self.anchor_mentions)
        return payload


@dataclass(frozen=True)
class EvidenceNeedBinding:
    binding_id: str
    assignments: Tuple[Tuple[str, str], ...] = ()
    prior: float = 1.0

    def assignment_map(self) -> Dict[str, str]:
        return {str(key): str(value) for key, value in self.assignments}

    def to_trace(self) -> Dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "assignments": {str(key): str(value) for key, value in self.assignments},
            "prior": float(self.prior),
        }


MAX_EVIDENCE_NEED_BINDINGS = 25
SUPPORT_MATRIX_SCHEMA_VERSION = "need_passage_support/v1"


def _coverage_for_requirement(scores: np.ndarray, positions: Sequence[int]) -> float:
    valid = [int(pos) for pos in positions if 0 <= int(pos) < scores.size]
    if not valid:
        return 0.0
    selected = np.clip(scores[valid], 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - selected))


def _support_matrix_rows(
    *,
    active_requirements: Sequence[EvidenceNeed],
    phi: np.ndarray,
    binding_idx: int,
    binding: EvidenceNeedBinding,
    pool_doc_titles: Sequence[str],
    baseline_positions: Sequence[int],
    final_positions: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_set = {int(pos) for pos in baseline_positions}
    final_set = {int(pos) for pos in final_positions}
    assignments = binding.assignment_map()
    for req_idx, requirement in enumerate(active_requirements):
        scores = np.clip(phi[req_idx, binding_idx, :], 0.0, 1.0)
        baseline_coverage = _coverage_for_requirement(scores, baseline_positions)
        final_coverage = _coverage_for_requirement(scores, final_positions)
        for pos, title in enumerate(pool_doc_titles[: scores.size]):
            baseline_without = [item for item in baseline_positions if int(item) != int(pos)]
            final_without = [item for item in final_positions if int(item) != int(pos)]
            rows.append(
                {
                    "need_id": str(requirement.unit_id),
                    "passage_position": int(pos),
                    "passage_title": str(title),
                    "support_score": round(float(scores[pos]), 6),
                    "baseline_selected": bool(pos in baseline_set),
                    "final_selected": bool(pos in final_set),
                    "baseline_coverage_delta": round(
                        float(baseline_coverage - _coverage_for_requirement(scores, baseline_without))
                        if pos in baseline_set
                        else 0.0,
                        6,
                    ),
                    "final_coverage_delta": round(
                        float(final_coverage - _coverage_for_requirement(scores, final_without))
                        if pos in final_set
                        else 0.0,
                        6,
                    ),
                    "selected_binding_id": str(binding.binding_id),
                    "binding_assignment": assignments.get(str(requirement.unit_id)) or None,
                    "supporting_bindings": [],
                }
            )
    return rows


def _strip_think_tags(text: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", str(text or "")).strip()


def _extract_json_payload(text: str) -> Any | None:
    cleaned = _strip_think_tags(text)
    candidates: List[str] = []
    for open_char, close_char in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_char)
        end = cleaned.rfind(close_char)
        if start >= 0 and end > start:
            candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _coerce_string_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        key = normalize_structure_text(text)
        if not text or not key or key in seen:
            continue
        cleaned.append(text)
        seen.add(key)
    return tuple(cleaned)


def _normalize_satisfiable_by(value: Any) -> str:
    key = normalize_structure_text(value)
    if key in {"document", "doc", "passage", "evidence"}:
        return "document"
    if key in {"inference", "reasoning", "reader", "aggregation", "comparison"}:
        return "inference"
    return "unknown"


def parse_evidence_need_response(
    response_text: str,
    *,
    max_steps: int = 4,
) -> Tuple[List[EvidenceNeed], Dict[str, object]]:
    payload = _extract_json_payload(response_text)
    trace: Dict[str, object] = {
        "parse_succeeded": False,
        "parse_error": None,
        "raw_preview": str(response_text or "")[:500],
        "parsed_step_count": 0,
        "active_step_count": 0,
    }
    if payload is None:
        trace["parse_error"] = "json_parse_failed"
        return [], trace

    if isinstance(payload, dict):
        steps = payload.get("steps", payload.get("requirements", []))
    else:
        steps = payload
    if not isinstance(steps, list):
        trace["parse_error"] = "payload_not_list"
        return [], trace

    requirements: List[EvidenceNeed] = []
    used_ids: set[str] = set()
    trace["parsed_step_count"] = int(len(steps))
    for idx, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            continue
        subquery = str(
            raw_step.get("subquery")
            or raw_step.get("question")
            or raw_step.get("need")
            or ""
        ).strip()
        if not subquery:
            continue
        raw_id = str(
            raw_step.get("id")
            or raw_step.get("unit_id")
            or raw_step.get("step_id")
            or f"s{idx + 1}"
        ).strip()
        unit_id = normalize_structure_text(raw_id).replace(" ", "_") or f"s{idx + 1}"
        if unit_id in used_ids:
            unit_id = f"{unit_id}_{idx + 1}"
        used_ids.add(unit_id)
        requirements.append(
            EvidenceNeed(
                unit_id=unit_id,
                subquery=subquery,
                depends_on=tuple(
                    normalize_structure_text(dep).replace(" ", "_")
                    for dep in _coerce_string_list(raw_step.get("depends_on", raw_step.get("dependencies", [])))
                    if normalize_structure_text(dep)
                ),
                expected_answer_type=str(raw_step.get("expected_answer_type", "unknown") or "unknown").strip().lower(),
                anchor_mentions=_coerce_string_list(raw_step.get("anchor_mentions", raw_step.get("anchors", []))),
                role=str(raw_step.get("role", raw_step.get("type", "support")) or "support").strip().lower(),
                satisfiable_by=_normalize_satisfiable_by(raw_step.get("satisfiable_by", "unknown")),
            )
        )
        if len(requirements) >= max(1, int(max_steps)):
            break

    valid_ids = {req.unit_id for req in requirements}
    normalized_requirements: List[EvidenceNeed] = []
    for req in requirements:
        normalized_requirements.append(
            EvidenceNeed(
                unit_id=req.unit_id,
                subquery=req.subquery,
                depends_on=tuple(dep for dep in req.depends_on if dep in valid_ids),
                expected_answer_type=req.expected_answer_type,
                anchor_mentions=req.anchor_mentions,
                role=req.role,
                satisfiable_by=req.satisfiable_by,
            )
        )

    trace["parse_succeeded"] = bool(normalized_requirements)
    trace["active_step_count"] = int(len(normalized_requirements))
    if not normalized_requirements:
        trace["parse_error"] = "no_active_steps"
    return normalized_requirements, trace


def build_fallback_evidence_needs(query: str) -> List[EvidenceNeed]:
    return [
        EvidenceNeed(
            unit_id="q0",
            subquery=str(query or "").strip(),
            depends_on=(),
            expected_answer_type="unknown",
            anchor_mentions=(),
            role="query",
            satisfiable_by="unknown",
        )
    ] if str(query or "").strip() else []


def _normalize_vector(vector: Any) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def _min_max(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    finite = np.asarray(values, dtype=float)
    finite[~np.isfinite(finite)] = float(np.nan)
    if np.all(np.isnan(finite)):
        return np.zeros_like(values, dtype=float)
    min_value = float(np.nanmin(finite))
    max_value = float(np.nanmax(finite))
    if max_value <= min_value:
        return np.ones_like(values, dtype=float) if max_value > 0 else np.zeros_like(values, dtype=float)
    normalized = (finite - min_value) / (max_value - min_value)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(float)


def _doc_text_body(doc_text: str) -> str:
    return str(doc_text or "").split("\n", 1)[1] if "\n" in str(doc_text or "") else str(doc_text or "")


def _anchor_hit_score(anchor_mentions: Sequence[str],
                      doc_title: str,
                      doc_text: str,
                      doc_entities: set[str]) -> float:
    if not anchor_mentions:
        return 0.0
    normalized_title = normalize_structure_text(doc_title)
    normalized_text = normalize_structure_text(_doc_text_body(doc_text))
    normalized_entities = {normalize_structure_text(entity) for entity in doc_entities}
    for anchor in anchor_mentions:
        key = normalize_structure_text(anchor)
        if not key:
            continue
        if key in normalized_title or normalized_title in key:
            return 1.0
        if key in normalized_entities:
            return 1.0
        if key in normalized_text:
            return 0.5
    return 0.0


def _dependency_hit_score(dependency_entities: set[str], doc_entities: set[str], doc_title: str) -> float:
    if not dependency_entities:
        return 0.0
    normalized_title = normalize_structure_text(doc_title)
    normalized_doc_entities = {normalize_structure_text(entity) for entity in doc_entities}
    overlap = dependency_entities & normalized_doc_entities
    if overlap:
        return 1.0
    for entity in dependency_entities:
        if entity and (entity in normalized_title or normalized_title in entity):
            return 0.75
    return 0.0


def _contains_normalized_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def _phrase_occurrences(haystack: str, needle: str) -> Tuple[int, int]:
    if not haystack or not needle:
        return 0, 10**9
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    matches = list(re.finditer(pattern, haystack))
    if not matches:
        return 0, 10**9
    return len(matches), int(matches[0].start())


def _is_generic_binding_title(title: str) -> bool:
    key = normalize_structure_text(title)
    if not key:
        return True
    generic_prefixes = (
        "list of ",
        "outline of ",
        "index of ",
        "timeline of ",
        "category ",
    )
    if any(key.startswith(prefix) for prefix in generic_prefixes):
        return True
    if len(key) <= 2:
        return True
    return False


def _candidate_type_compatible(title: str, expected_answer_type: str) -> bool:
    if _is_generic_binding_title(title):
        return False
    expected = normalize_structure_text(expected_answer_type)
    if not expected or expected in {"unknown", "entity", "thing"}:
        return True
    key = normalize_structure_text(title)
    # Keep this conservative: only reject obvious document-type mismatches.
    work_markers = {" film", " song", " album", " episode", " season", " novel", " book"}
    if expected in {"person", "people", "human"}:
        return not any(marker in f" {key}" for marker in {" film", " song", " album", " episode", " season"})
    if expected in {"location", "place", "country", "city"}:
        return not any(marker in f" {key}" for marker in work_markers)
    if expected in {"organization", "organisation", "company", "team"}:
        return not any(marker in f" {key}" for marker in work_markers)
    return True


_GUARDED_SUBSTRING_SINGLE_TOKEN_BLOCKLIST = {
    # Country / demonym-like single-token bridge entities are too ambiguous for
    # substring title binding (e.g. "France" -> "Rudolph of France").
    "afghanistan",
    "albania",
    "algeria",
    "andorra",
    "angola",
    "argentina",
    "armenia",
    "australia",
    "austria",
    "azerbaijan",
    "bahamas",
    "bahrain",
    "bangladesh",
    "barbados",
    "belarus",
    "belgium",
    "belize",
    "benin",
    "bhutan",
    "bolivia",
    "botswana",
    "brazil",
    "bulgaria",
    "burundi",
    "cambodia",
    "cameroon",
    "canada",
    "chad",
    "chile",
    "china",
    "colombia",
    "comoros",
    "croatia",
    "cuba",
    "cyprus",
    "denmark",
    "djibouti",
    "dominica",
    "ecuador",
    "egypt",
    "eritrea",
    "estonia",
    "ethiopia",
    "fiji",
    "finland",
    "france",
    "gabon",
    "gambia",
    "georgia",
    "germany",
    "ghana",
    "greece",
    "grenada",
    "guatemala",
    "guinea",
    "guyana",
    "haiti",
    "honduras",
    "hungary",
    "iceland",
    "india",
    "indonesia",
    "iran",
    "iraq",
    "ireland",
    "israel",
    "italy",
    "jamaica",
    "japan",
    "jordan",
    "kazakhstan",
    "kenya",
    "kiribati",
    "kuwait",
    "kyrgyzstan",
    "laos",
    "latvia",
    "lebanon",
    "lesotho",
    "liberia",
    "libya",
    "liechtenstein",
    "lithuania",
    "luxembourg",
    "madagascar",
    "malawi",
    "malaysia",
    "maldives",
    "mali",
    "malta",
    "mauritania",
    "mauritius",
    "mexico",
    "moldova",
    "monaco",
    "mongolia",
    "montenegro",
    "morocco",
    "mozambique",
    "myanmar",
    "namibia",
    "nauru",
    "nepal",
    "netherlands",
    "nicaragua",
    "niger",
    "nigeria",
    "norway",
    "oman",
    "pakistan",
    "palau",
    "panama",
    "paraguay",
    "peru",
    "philippines",
    "poland",
    "portugal",
    "qatar",
    "romania",
    "russia",
    "rwanda",
    "samoa",
    "senegal",
    "serbia",
    "seychelles",
    "singapore",
    "slovakia",
    "slovenia",
    "somalia",
    "spain",
    "sudan",
    "suriname",
    "sweden",
    "switzerland",
    "syria",
    "tajikistan",
    "tanzania",
    "thailand",
    "togo",
    "tonga",
    "tunisia",
    "turkey",
    "turkmenistan",
    "tuvalu",
    "uganda",
    "ukraine",
    "uruguay",
    "uzbekistan",
    "vanuatu",
    "venezuela",
    "vietnam",
    "yemen",
    "zambia",
    "zimbabwe",
    # High-risk single-token personal/common names seen in audit failures.
    "muhammad",
}


def _substring_title_match_is_guarded(entity: str, title: str) -> bool:
    entity_key = normalize_structure_text(entity)
    title_key = normalize_structure_text(title)
    if not entity_key or not title_key:
        return False
    if entity_key == title_key:
        return True
    if entity_key.isdigit() or title_key.isdigit():
        return False

    entity_tokens = entity_key.split()
    title_tokens = title_key.split()
    if not entity_tokens or not title_tokens:
        return False
    if len(entity_tokens) == 1 and entity_tokens[0] in _GUARDED_SUBSTRING_SINGLE_TOKEN_BLOCKLIST:
        return False

    def has_contiguous_subsequence(shorter: List[str], longer: List[str]) -> bool:
        if len(shorter) > len(longer):
            return False
        width = len(shorter)
        return any(longer[start:start + width] == shorter for start in range(len(longer) - width + 1))

    # Prefer entity -> title containment. Reverse containment is allowed only
    # when the matched title phrase has at least two tokens, avoiding cases like
    # a long extracted phrase binding to a one-token generic page.
    if has_contiguous_subsequence(entity_tokens, title_tokens):
        return True
    if len(title_tokens) >= 2 and has_contiguous_subsequence(title_tokens, entity_tokens):
        return True
    return False


def _remove_parenthetical_disambiguation(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(text or "")).strip()


def _contiguous_token_subsequence(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    if not shorter or len(shorter) > len(longer):
        return False
    width = len(shorter)
    return any(list(longer[start:start + width]) == list(shorter) for start in range(len(longer) - width + 1))


def _wiki_title_match_type(entity: str, title: str) -> str | None:
    entity_key = normalize_structure_text(entity)
    title_key = normalize_structure_text(title)
    title_base_key = normalize_structure_text(_remove_parenthetical_disambiguation(title))
    if not entity_key or not title_key:
        return None
    if entity_key == title_key:
        return "exact"
    if entity_key.isdigit():
        return None
    if title_base_key and entity_key == title_base_key:
        return "disambiguation"

    entity_tokens = entity_key.split()
    title_tokens = title_key.split()
    if len(entity_tokens) >= 2 and _contiguous_token_subsequence(entity_tokens, title_tokens):
        return "multi_token_alias"
    return None


def _title_match_pool_position_and_type(
    entity: str,
    pool_titles: Sequence[str],
    *,
    allow_substring: bool = True,
    guarded_substring: bool = False,
    wiki_title_match: bool = False,
    require_unique_wiki_title_match: bool = False,
) -> Tuple[int | None, str]:
    entity_lower = entity.lower().strip()
    if not entity_lower:
        return None, "none"
    for idx, title in enumerate(pool_titles):
        if entity_lower == title.lower().strip():
            return idx, "exact"
    if wiki_title_match:
        wiki_matches: List[Tuple[int, str]] = []
        for idx, title in enumerate(pool_titles):
            match_type = _wiki_title_match_type(entity, str(title))
            if match_type and match_type != "exact":
                wiki_matches.append((idx, match_type))
        if len(wiki_matches) == 1:
            return wiki_matches[0]
        if wiki_matches and not require_unique_wiki_title_match:
            return wiki_matches[0]
        return None, "none"
    if allow_substring:
        for idx, title in enumerate(pool_titles):
            tl = title.lower().strip()
            if not ((entity_lower in tl or tl in entity_lower) and min(len(entity_lower), len(tl)) >= 3):
                continue
            if guarded_substring and not _substring_title_match_is_guarded(entity, str(title)):
                continue
            return idx, "substring"
    return None, "none"


def _title_match_pool_position(
    entity: str,
    pool_titles: Sequence[str],
    *,
    allow_substring: bool = True,
    guarded_substring: bool = False,
    wiki_title_match: bool = False,
    require_unique_wiki_title_match: bool = False,
) -> int | None:
    idx, _ = _title_match_pool_position_and_type(
        entity,
        pool_titles,
        allow_substring=allow_substring,
        guarded_substring=guarded_substring,
        wiki_title_match=wiki_title_match,
        require_unique_wiki_title_match=require_unique_wiki_title_match,
    )
    return idx


def _binding_candidate_hit_score(candidate_key: str,
                                 doc_entities: set[str],
                                 doc_title: str,
                                 doc_text: str) -> float:
    if not candidate_key:
        return 0.0
    title_key = normalize_structure_text(doc_title)
    if candidate_key and (candidate_key == title_key or candidate_key in title_key or title_key in candidate_key):
        return 1.0
    normalized_entities = {normalize_structure_text(entity) for entity in doc_entities}
    if candidate_key in normalized_entities:
        return 1.0
    normalized_text = normalize_structure_text(_doc_text_body(doc_text))
    if _contains_normalized_phrase(normalized_text, candidate_key):
        return 0.5
    return 0.0


def _llm_binding_compat_score(candidate_title: str,
                              doc_title: str,
                              doc_text: str,
                              *,
                              body_mention_weight: float = 0.0) -> float:
    candidate_key = normalize_structure_text(candidate_title)
    if not candidate_key:
        return 0.0
    if _title_match_pool_position(candidate_title, [doc_title]) is not None:
        return 1.0
    body_weight = min(max(float(body_mention_weight), 0.0), 1.0)
    if body_weight <= 0.0:
        return 0.0
    normalized_body = normalize_structure_text(_doc_text_body(doc_text))
    if _contains_normalized_phrase(normalized_body, candidate_key):
        return body_weight
    return 0.0


def _binding_title_resolution_stats(
    binding_candidates_by_req: Mapping[str, Sequence[Mapping[str, object]]],
    pool_doc_titles: Sequence[str],
) -> Dict[str, object]:
    all_candidates: List[Mapping[str, object]] = []
    for rows in binding_candidates_by_req.values():
        all_candidates.extend(row for row in rows if isinstance(row, Mapping))

    pool_title_counts = Counter(
        normalize_structure_text(title)
        for title in pool_doc_titles
        if normalize_structure_text(title)
    )
    title_occurrences = [
        int(pool_title_counts.get(normalize_structure_text(row.get("title", "")), 0))
        for row in all_candidates
        if normalize_structure_text(row.get("title", ""))
    ]
    candidate_count = int(len(all_candidates))
    unique_count = int(sum(1 for count in title_occurrences if count == 1))
    nonunique_count = int(sum(1 for count in title_occurrences if count > 1))
    title_unique_rate = float(unique_count) / float(candidate_count) if candidate_count else 0.0
    title_nonunique_rate = float(nonunique_count) / float(candidate_count) if candidate_count else 0.0
    avg_occurrences = float(np.mean(title_occurrences)) if title_occurrences else 0.0
    return {
        "candidate_count": candidate_count,
        "candidate_requirement_count": int(sum(1 for rows in binding_candidates_by_req.values() if rows)),
        "unique_candidate_title_count": unique_count,
        "nonunique_candidate_title_count": nonunique_count,
        "title_unique_rate": round(float(title_unique_rate), 6),
        "title_nonunique_rate": round(float(title_nonunique_rate), 6),
        "avg_candidate_title_occurrences": round(float(avg_occurrences), 6),
    }


def _build_bound_subquery(subquery: str, candidate_title: str) -> str:
    text = str(subquery or "").strip()
    title = str(candidate_title or "").strip()
    if not text or not title:
        return text
    replaced = re.sub(r"\[[^\]]+\]", title, text)
    referent_pattern = re.compile(
        r"\b(that|the)\s+("
        r"person|entity|director|performer|singer|composer|author|writer|actor|actress|"
        r"father|mother|parent|husband|wife|spouse|child|team|city|place|country|"
        r"organization|organisation|company|film|movie|work"
        r")\b",
        flags=re.IGNORECASE,
    )
    replaced = referent_pattern.sub(title, replaced, count=1)
    if replaced != text:
        return replaced
    return f"{text} Target entity: {title}."


_INFERENCE_ONLY_PATTERN = re.compile(
    r"\b("
    r"same|both|either|neither|compare|comparison|"
    r"earlier|later|older|younger|first|last|"
    r"more|less|larger|smaller|highest|lowest|"
    r"which\s+.+\s+(?:first|earlier|later|older|younger)"
    r")\b",
    flags=re.IGNORECASE,
)


def _normalize_satisfiable_by_policy(value: str | None) -> str:
    policy = str(value or "binding_override").strip().lower().replace("-", "_")
    if policy in {"strict", "binding_override", "grounded_override", "regex_only"}:
        return policy
    return "binding_override"


def _is_inference_only_requirement(
    req: EvidenceNeed,
    *,
    use_satisfiable_by: bool = True,
) -> bool:
    """Return True for final comparison/aggregation needs that a selection cannot satisfy directly."""
    if bool(use_satisfiable_by):
        satisfiable_by = _normalize_satisfiable_by(req.satisfiable_by)
        if satisfiable_by == "inference":
            return True
        if satisfiable_by == "document":
            return False
    role = normalize_structure_text(req.role)
    if role in {"comparison", "compare", "comparator"}:
        return True
    if role != "answer":
        return False
    text = f"{req.subquery} {req.expected_answer_type}"
    return bool(_INFERENCE_ONLY_PATTERN.search(text))


def _requirement_inference_veto(
    req: EvidenceNeed,
    *,
    has_explicit_anchor: bool,
    has_resolved_binding: bool,
    satisfiable_by_policy: str,
) -> Tuple[bool, bool]:
    policy = _normalize_satisfiable_by_policy(satisfiable_by_policy)
    field_or_regex_veto = _is_inference_only_requirement(req, use_satisfiable_by=True)
    regex_veto = _is_inference_only_requirement(req, use_satisfiable_by=False)
    if policy == "regex_only":
        return bool(regex_veto), bool(regex_veto)
    if policy == "strict":
        return bool(field_or_regex_veto), bool(field_or_regex_veto)
    if policy == "grounded_override":
        effective_veto = bool(field_or_regex_veto and not (has_explicit_anchor or has_resolved_binding))
        return bool(field_or_regex_veto), effective_veto
    effective_veto = bool(field_or_regex_veto and not has_resolved_binding)
    return bool(field_or_regex_veto), effective_veto


def _is_operator_evidence_need(req: EvidenceNeed) -> bool:
    role = normalize_structure_text(req.role)
    if role in {"comparison", "compare", "comparator", "aggregation", "aggregate"}:
        return True
    return _normalize_satisfiable_by(req.satisfiable_by) == "inference"


def _compute_embedding_match_scores(
    requirements: Sequence[EvidenceNeed],
    requirement_embeddings: Dict[str, np.ndarray],
    doc_vectors: Dict[int, np.ndarray],
    pool_limit: int,
) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    for req in requirements:
        req_vec = _normalize_vector(requirement_embeddings.get(req.unit_id, np.array([])))
        values: List[float] = []
        for pos in range(pool_limit):
            doc_vec = doc_vectors.get(pos)
            if req_vec.size == 0 or doc_vec is None or req_vec.shape != doc_vec.shape:
                values.append(float("-inf"))
            else:
                values.append(float(np.dot(req_vec, doc_vec)))
        scores[req.unit_id] = _min_max(np.asarray(values, dtype=float))
    return scores


def _coverage_noisy_or(phi_for_binding: np.ndarray, selected_positions: Sequence[int]) -> np.ndarray:
    if phi_for_binding.size == 0:
        return np.asarray([], dtype=float)
    valid_positions = [
        int(pos)
        for pos in selected_positions
        if 0 <= int(pos) < phi_for_binding.shape[1]
    ]
    if not valid_positions:
        return np.zeros(phi_for_binding.shape[0], dtype=float)
    selected_phi = np.clip(phi_for_binding[:, valid_positions], 0.0, 1.0)
    return np.clip(1.0 - np.prod(1.0 - selected_phi, axis=1), 0.0, 1.0)


def _coverage_rank_scores(pool_doc_scores: Sequence[float] | None, pool_limit: int) -> np.ndarray:
    if pool_limit <= 0:
        return np.asarray([], dtype=float)
    if pool_doc_scores is None:
        return np.linspace(1.0, 0.0, pool_limit, dtype=float)
    try:
        values = np.asarray(list(pool_doc_scores)[:pool_limit], dtype=float).reshape(-1)
    except Exception:
        values = np.asarray([], dtype=float)
    if values.size < pool_limit:
        fallback = np.linspace(float(pool_limit), 1.0, pool_limit, dtype=float)
        if values.size:
            fallback[: values.size] = values
        values = fallback
    values = values[:pool_limit]
    if not np.any(np.isfinite(values)):
        return np.linspace(1.0, 0.0, pool_limit, dtype=float)
    return _min_max(values)


def _replace_at_position(positions: Sequence[int], old_pos: int, new_pos: int) -> List[int]:
    updated: List[int] = []
    seen: set[int] = set()
    for pos in positions:
        candidate = int(new_pos) if int(pos) == int(old_pos) else int(pos)
        if candidate in seen:
            continue
        updated.append(candidate)
        seen.add(candidate)
    return updated


def _coerce_optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raw_values = [values]
    elif isinstance(values, Sequence):
        raw_values = list(values)
    else:
        raw_values = [values]
    coerced: List[int] = []
    seen: set[int] = set()
    for value in raw_values:
        int_value = _coerce_optional_int(value)
        if int_value is None or int_value in seen:
            continue
        coerced.append(int_value)
        seen.add(int_value)
    return coerced


def _positions_for_doc_ids(
    *,
    pool_doc_ids: Sequence[int | None],
    target_doc_ids: Sequence[Any],
    pool_limit: int,
) -> List[int]:
    position_by_doc_id: Dict[int, int] = {}
    limit = max(0, min(int(pool_limit), len(pool_doc_ids)))
    for pos, raw_doc_id in enumerate(pool_doc_ids[:limit]):
        doc_id = _coerce_optional_int(raw_doc_id)
        if doc_id is None:
            continue
        position_by_doc_id.setdefault(int(doc_id), int(pos))

    positions: List[int] = []
    seen: set[int] = set()
    for raw_doc_id in target_doc_ids:
        doc_id = _coerce_optional_int(raw_doc_id)
        if doc_id is None:
            continue
        pos = position_by_doc_id.get(int(doc_id))
        if pos is None or pos in seen:
            continue
        positions.append(int(pos))
        seen.add(int(pos))
    return positions


def _agreement_retention_scores(
    *,
    pool_doc_ids: Sequence[int | None],
    pool_limit: int,
    baseline_positions: Sequence[int],
    rebuild_positions: Sequence[int],
    evidence_link_metadata: Mapping[str, Any] | None,
    include_coverage_signal: bool,
    agreement_top_k: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Count final-budget-aligned corroboration votes for each pool position."""
    limit = max(0, min(int(pool_limit), len(pool_doc_ids)))
    scores = np.zeros(limit, dtype=float)
    top_k = max(1, int(agreement_top_k))
    signal_positions: Dict[str, List[int]] = {}

    def add_signal(name: str, positions: Sequence[int]) -> None:
        clean_positions: List[int] = []
        seen: set[int] = set()
        for raw_pos in positions:
            pos = _coerce_optional_int(raw_pos)
            if pos is None or pos < 0 or pos >= limit or pos in seen:
                continue
            clean_positions.append(int(pos))
            seen.add(int(pos))
            scores[int(pos)] += 1.0
            if len(clean_positions) >= top_k:
                break
        signal_positions[name] = clean_positions

    add_signal("baseline_topk", list(baseline_positions)[:top_k])

    dense_doc_ids: List[int] = []
    if isinstance(evidence_link_metadata, Mapping):
        dense_doc_ids = _coerce_int_list(evidence_link_metadata.get("source_prior_prefix_doc_indices"))
        if not dense_doc_ids:
            dense_doc_ids = _coerce_int_list(evidence_link_metadata.get("native_dense_doc_indices"))
    dense_positions = _positions_for_doc_ids(
        pool_doc_ids=pool_doc_ids,
        target_doc_ids=dense_doc_ids[:top_k],
        pool_limit=limit,
    )
    if not dense_positions:
        dense_positions = list(range(min(top_k, limit)))
    add_signal("dense_topk", dense_positions)

    if include_coverage_signal:
        add_signal("coverage_topk", list(rebuild_positions)[:top_k])

    return scores, {
        "agreement_top_k": int(top_k),
        "include_coverage_signal": bool(include_coverage_signal),
        "signal_positions": signal_positions,
        "signal_count": int(len(signal_positions)),
        "score_histogram": {
            str(value): int(np.sum(scores == float(value)))
            for value in sorted(set(float(item) for item in scores.tolist()))
        },
    }


def _build_evidence_link_graph_prior(
    *,
    evidence_link_metadata: Mapping[str, Any] | None,
    pool_doc_ids: Sequence[int | None],
    pool_limit: int,
    w_selected: float = 1.0,
    w_anchor: float = 0.3,
    w_rank: float = 0.2,
    dense_anchor_top_k: int = 20,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build a per-document EvLink prior aligned to the current coverage-aware selection pool."""
    limit = max(0, min(int(pool_limit), len(pool_doc_ids)))
    prior = np.zeros(limit, dtype=float)
    trace: Dict[str, object] = {
        "metadata_present": bool(evidence_link_metadata),
        "component_weights": {
            "selected": round(float(w_selected), 6),
            "anchor": round(float(w_anchor), 6),
            "rank": round(float(w_rank), 6),
        },
        "dense_anchor_top_k": int(max(0, dense_anchor_top_k)),
        "selected_positions": [],
        "dense_anchor_positions": [],
        "retrieved_positions": [],
        "stats": {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "nonzero_count": 0,
        },
    }
    if not evidence_link_metadata or limit <= 0:
        return prior, trace

    selected_doc_indices = _coerce_int_list(evidence_link_metadata.get("selected_doc_indices"))
    selected_evidence_set = evidence_link_metadata.get("selected_evidence_set")
    if not selected_doc_indices and isinstance(selected_evidence_set, Mapping):
        selected_doc_indices = _coerce_int_list(selected_evidence_set.get("doc_indices"))
    selected_ids = set(selected_doc_indices)

    dense_anchor_doc_indices = _coerce_int_list(evidence_link_metadata.get("native_dense_doc_indices"))
    anchor_keep = max(0, int(dense_anchor_top_k))
    dense_anchor_ids = set(dense_anchor_doc_indices[:anchor_keep] if anchor_keep else [])

    retrieved_doc_indices = _coerce_int_list(evidence_link_metadata.get("retrieved_doc_indices"))
    rank_score_by_doc_id: Dict[int, float] = {}
    if retrieved_doc_indices:
        denom = float(max(len(retrieved_doc_indices) - 1, 1))
        for rank, doc_id in enumerate(retrieved_doc_indices):
            rank_score_by_doc_id.setdefault(int(doc_id), 1.0 - float(rank) / denom)

    positive_weights = [
        max(float(w_selected), 0.0),
        max(float(w_anchor), 0.0),
        max(float(w_rank), 0.0),
    ]
    weight_denom = float(sum(positive_weights))
    if weight_denom <= 1e-12:
        return prior, trace

    selected_positions: List[int] = []
    dense_anchor_positions: List[int] = []
    retrieved_positions: List[int] = []
    for pos, raw_doc_id in enumerate(pool_doc_ids[:limit]):
        doc_id = _coerce_optional_int(raw_doc_id)
        if doc_id is None:
            continue
        score = 0.0
        if positive_weights[0] > 0.0 and doc_id in selected_ids:
            score += positive_weights[0]
            selected_positions.append(int(pos))
        if positive_weights[1] > 0.0 and doc_id in dense_anchor_ids:
            score += positive_weights[1]
            dense_anchor_positions.append(int(pos))
        if positive_weights[2] > 0.0:
            rank_score = float(rank_score_by_doc_id.get(doc_id, 0.0))
            if rank_score > 0.0:
                retrieved_positions.append(int(pos))
            score += positive_weights[2] * rank_score
        prior[int(pos)] = min(max(score / weight_denom, 0.0), 1.0)

    trace["selected_positions"] = selected_positions
    trace["dense_anchor_positions"] = dense_anchor_positions
    trace["retrieved_positions"] = retrieved_positions
    if prior.size:
        trace["stats"] = {
            "min": round(float(np.min(prior)), 6),
            "max": round(float(np.max(prior)), 6),
            "mean": round(float(np.mean(prior)), 6),
            "nonzero_count": int(np.sum(prior > 0.0)),
        }
    return prior, trace


def _binding_grounding_scores(
    *,
    phi: np.ndarray,
    active_requirements: Sequence[EvidenceNeed],
    bindings: Sequence[EvidenceNeedBinding],
    binding_assignment_rows_by_idx: Sequence[Sequence[Mapping[str, object]]],
    pool_doc_titles: Sequence[str],
    pool_limit: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Score each binding by upstream phi support for its assigned entity docs."""
    scores = np.ones(len(bindings), dtype=float)
    traces: List[Dict[str, object]] = []
    req_index_by_id = {str(req.unit_id): int(idx) for idx, req in enumerate(active_requirements)}
    title_limit = list(pool_doc_titles[:pool_limit])

    for binding_idx, binding in enumerate(bindings):
        rows = list(binding_assignment_rows_by_idx[binding_idx]) if binding_idx < len(binding_assignment_rows_by_idx) else []
        assignment_scores: List[float] = []
        assignment_traces: List[Dict[str, object]] = []
        for row in rows:
            req_id = str(row.get("requirement_id", "") or "")
            dep_id = str(row.get("dep", "") or "")
            title = str(row.get("title", "") or "")
            try:
                entity_pos = int(row.get("title_pool_position", -1))
            except (TypeError, ValueError):
                entity_pos = -1
            if entity_pos < 0 or entity_pos >= pool_limit:
                entity_pos = _title_match_pool_position(title, title_limit)
                entity_pos = int(entity_pos) if entity_pos is not None else -1

            upstream_idx = req_index_by_id.get(dep_id)
            grounding = 0.0
            if upstream_idx is not None and 0 <= entity_pos < pool_limit and 0 <= binding_idx < phi.shape[1]:
                grounding = float(phi[int(upstream_idx), int(binding_idx), int(entity_pos)])
            assignment_scores.append(float(grounding))
            assignment_traces.append({
                "requirement_id": req_id,
                "dep": dep_id,
                "title": title,
                "title_pool_position": int(entity_pos),
                "upstream_requirement_index": int(upstream_idx) if upstream_idx is not None else None,
                "grounding": round(float(grounding), 6),
            })

        score = float(np.mean(assignment_scores)) if assignment_scores else 1.0
        scores[int(binding_idx)] = min(max(score, 0.0), 1.0)
        traces.append({
            "binding_id": str(binding.binding_id),
            "score": round(float(scores[int(binding_idx)]), 6),
            "assignments": assignment_traces,
        })
    return scores, traces


def _refine_by_single_swaps(
    *,
    binding_idx: int,
    initial_positions: Sequence[int],
    pool_limit: int,
    objective_fn: Callable[[int, Sequence[int]], Tuple[float, np.ndarray]],
    min_gain: float = 0.001,
    max_passes: int = 1,
) -> Tuple[List[int], float, np.ndarray, List[Dict[str, object]]]:
    """Run a small 1-swap local search over the fixed-binding coverage-aware selection objective."""
    current_positions = list(dict.fromkeys(int(pos) for pos in initial_positions if 0 <= int(pos) < pool_limit))
    current_score, current_coverage = objective_fn(int(binding_idx), current_positions)
    steps: List[Dict[str, object]] = []
    max_pass_count = max(0, int(max_passes))
    min_delta = max(float(min_gain), 0.0)
    for _ in range(max_pass_count):
        best_swap: Dict[str, object] | None = None
        current_set = set(current_positions)
        for out_pos in list(current_positions):
            for in_pos in range(pool_limit):
                if int(in_pos) in current_set:
                    continue
                proposed = _replace_at_position(current_positions, int(out_pos), int(in_pos))
                proposed_score, proposed_coverage = objective_fn(int(binding_idx), proposed)
                gain = float(proposed_score - current_score)
                if gain <= min_delta:
                    continue
                row = {
                    "out_position": int(out_pos),
                    "in_position": int(in_pos),
                    "objective_gain": gain,
                    "objective": float(proposed_score),
                    "coverage": proposed_coverage,
                }
                if best_swap is None or (
                    float(row["objective_gain"]),
                    float(row["objective"]),
                    -int(row["in_position"]),
                ) > (
                    float(best_swap["objective_gain"]),
                    float(best_swap["objective"]),
                    -int(best_swap["in_position"]),
                ):
                    best_swap = row
        if best_swap is None:
            break
        current_positions = _replace_at_position(
            current_positions,
            int(best_swap["out_position"]),
            int(best_swap["in_position"]),
        )
        current_score = float(best_swap["objective"])
        current_coverage = np.asarray(best_swap["coverage"], dtype=float)
        steps.append({
            "step": int(len(steps) + 1),
            "mode": "coverage_noisy_or_single_admission",
            "out_position": int(best_swap["out_position"]),
            "in_position": int(best_swap["in_position"]),
            "objective_gain": round(float(best_swap["objective_gain"]), 6),
            "objective": round(float(current_score), 6),
        })
    return current_positions, float(current_score), np.asarray(current_coverage, dtype=float), steps


def _topological_requirement_order(requirements: Sequence[EvidenceNeed]) -> List[EvidenceNeed]:
    req_by_id = {str(req.unit_id): req for req in requirements}
    visited: set[str] = set()
    visiting: set[str] = set()
    ordered: List[EvidenceNeed] = []

    def visit(req: EvidenceNeed) -> None:
        req_id = str(req.unit_id)
        if req_id in visited:
            return
        if req_id in visiting:
            return
        visiting.add(req_id)
        for dep_id in req.depends_on:
            dep = req_by_id.get(str(dep_id))
            if dep is not None:
                visit(dep)
        visiting.discard(req_id)
        visited.add(req_id)
        ordered.append(req)

    for requirement in requirements:
        visit(requirement)
    return ordered


def select_minimal_demand_repair_positions(
    *,
    query: str,
    requirements: Sequence[EvidenceNeed],
    requirement_embeddings: Dict[str, np.ndarray],
    pool_docs: Sequence[str],
    pool_doc_ids: Sequence[int | None],
    pool_doc_titles: Sequence[str],
    passage_embeddings: np.ndarray,
    qa_top_k: int,
    tau_percentile: float = 50.0,
    edit_budget: int = 2,
) -> Tuple[List[int], Dict[str, object]]:
    """Repair unmet demand coverage using only cosine phi.

    This intentionally excludes anchor, binding, relation, sentence-level, and
    retriever-prior features. It is a minimal sanity probe for the repair
    framing rather than a tuned coverage-aware selection variant.
    """
    selection_label = "minimal_demand_repair"
    pool_limit = len(pool_docs)
    target_k = min(max(int(qa_top_k), 0), pool_limit)
    if target_k <= 0:
        return [], {"selection": selection_label, "status": "empty_pool"}

    baseline_positions = list(range(target_k))
    active_requirements = [
        req for req in requirements
        if str(req.subquery or "").strip() and not _is_operator_evidence_need(req)
    ]
    if not active_requirements:
        return baseline_positions, {
            "selection": selection_label,
            "status": "fallback_no_requirements",
            "query": str(query),
            "selected_positions": baseline_positions,
            "keep_baseline": True,
            "edit_count": 0,
            "selection_steps": [],
        }

    doc_vectors: Dict[int, np.ndarray] = {}
    for pos, doc_id in enumerate(pool_doc_ids[:pool_limit]):
        if doc_id is None or int(doc_id) < 0 or int(doc_id) >= len(passage_embeddings):
            continue
        vector = _normalize_vector(passage_embeddings[int(doc_id)])
        if vector.size:
            doc_vectors[int(pos)] = vector
    phi_by_req = _compute_embedding_match_scores(
        active_requirements,
        requirement_embeddings,
        doc_vectors,
        pool_limit,
    )
    phi = np.asarray(
        [phi_by_req.get(req.unit_id, np.zeros(pool_limit, dtype=float)) for req in active_requirements],
        dtype=float,
    )
    finite_phi = phi[np.isfinite(phi)]
    tau = float(np.percentile(finite_phi, float(tau_percentile))) if finite_phi.size else 1.0
    tau = float(np.clip(tau, 0.0, 1.0))

    req_index_by_id = {str(req.unit_id): idx for idx, req in enumerate(active_requirements)}
    baseline_max_by_req = {
        str(req.unit_id): float(np.max(phi[idx, baseline_positions])) if baseline_positions else 0.0
        for idx, req in enumerate(active_requirements)
    }
    unmet_ids = {
        str(req_id)
        for req_id, score in baseline_max_by_req.items()
        if float(score) < tau
    }
    if not unmet_ids:
        return baseline_positions, {
            "selection": selection_label,
            "status": "keep_all_demands_met",
            "query": str(query),
            "requirement_count": int(len(active_requirements)),
            "requirements": [req.to_trace() for req in active_requirements],
            "tau_percentile": round(float(tau_percentile), 4),
            "tau": round(float(tau), 6),
            "baseline_max_by_requirement": {
                req_id: round(float(score), 6) for req_id, score in baseline_max_by_req.items()
            },
            "unmet_requirement_ids": [],
            "selected_positions": baseline_positions,
            "selected_titles": [pool_doc_titles[pos] for pos in baseline_positions],
            "keep_baseline": True,
            "edit_count": 0,
            "edit_budget": int(max(0, edit_budget)),
            "selection_steps": [],
        }

    def coverage_loss_count(out_pos: int, positions: Sequence[int]) -> int:
        selected = [int(pos) for pos in positions if 0 <= int(pos) < pool_limit]
        if int(out_pos) not in selected:
            return 0
        loss = 0
        for req_idx in range(len(active_requirements)):
            scores = [(pos, float(phi[req_idx, pos])) for pos in selected]
            scores.sort(key=lambda item: (-item[1], item[0]))
            if not scores or scores[0][0] != int(out_pos) or scores[0][1] < tau:
                continue
            second_score = scores[1][1] if len(scores) > 1 else 0.0
            if scores[0][1] > second_score + 1e-9:
                loss += 1
        return int(loss)

    selected_positions = list(baseline_positions)
    edit_limit = max(0, int(edit_budget))
    steps: List[Dict[str, object]] = []
    ordered_requirements = [
        req for req in _topological_requirement_order(active_requirements)
        if str(req.unit_id) in unmet_ids
    ]
    for req in ordered_requirements:
        if len(steps) >= edit_limit:
            break
        req_idx = req_index_by_id.get(str(req.unit_id))
        if req_idx is None:
            continue
        selected_set = set(selected_positions)
        candidates = [pos for pos in range(pool_limit) if pos not in selected_set]
        if not candidates:
            break
        in_pos = max(candidates, key=lambda pos: (float(phi[req_idx, pos]), -int(pos)))
        in_score = float(phi[req_idx, in_pos])
        if in_score < tau:
            continue
        out_pos = min(
            selected_positions,
            key=lambda pos: (coverage_loss_count(int(pos), selected_positions), int(pos)),
        )
        loss = coverage_loss_count(int(out_pos), selected_positions)
        if in_score <= float(loss):
            continue
        selected_positions = _replace_at_position(selected_positions, int(out_pos), int(in_pos))
        steps.append({
            "step": int(len(steps) + 1),
            "mode": "minimal_demand_repair",
            "requirement_id": str(req.unit_id),
            "requirement_subquery": str(req.subquery),
            "out_position": int(out_pos),
            "out_title": str(pool_doc_titles[int(out_pos)]),
            "in_position": int(in_pos),
            "in_title": str(pool_doc_titles[int(in_pos)]),
            "in_phi": round(float(in_score), 6),
            "coverage_loss_count": int(loss),
            "tau": round(float(tau), 6),
        })

    keep_baseline = list(selected_positions) == baseline_positions
    return selected_positions, {
        "selection": selection_label,
        "status": "keep_no_accepted_repair" if keep_baseline else "repaired",
        "query": str(query),
        "requirement_count": int(len(active_requirements)),
        "requirements": [req.to_trace() for req in active_requirements],
        "tau_percentile": round(float(tau_percentile), 4),
        "tau": round(float(tau), 6),
        "baseline_max_by_requirement": {
            req_id: round(float(score), 6) for req_id, score in baseline_max_by_req.items()
        },
        "unmet_requirement_ids": sorted(unmet_ids),
        "selected_positions": list(selected_positions),
        "selected_titles": [pool_doc_titles[pos] for pos in selected_positions],
        "baseline_positions": baseline_positions,
        "baseline_titles": [pool_doc_titles[pos] for pos in baseline_positions],
        "keep_baseline": bool(keep_baseline),
        "edit_count": int(len(steps)),
        "edit_budget": int(edit_limit),
        "selection_steps": steps,
    }


def select_minimal_demand_repair_nli_positions(
    *,
    query: str,
    requirements: Sequence[EvidenceNeed],
    pool_docs: Sequence[str],
    pool_doc_titles: Sequence[str],
    qa_top_k: int,
    nli_score_fn: Callable[[Sequence[Tuple[str, str]]], List[float]],
    tau_percentile: float = 50.0,
    edit_budget: int = 2,
) -> Tuple[List[int], Dict[str, object]]:
    """MDR with NLI entailment as phi(r, d) instead of cosine."""
    selection_label = "minimal_demand_repair_nli"
    pool_limit = len(pool_docs)
    target_k = min(max(int(qa_top_k), 0), pool_limit)
    if target_k <= 0:
        return [], {"selection": selection_label, "status": "empty_pool"}

    baseline_positions = list(range(target_k))
    active_requirements = [
        req for req in requirements
        if str(req.subquery or "").strip() and not _is_operator_evidence_need(req)
    ]
    if not active_requirements:
        return baseline_positions, {
            "selection": selection_label,
            "status": "fallback_no_requirements",
            "query": str(query),
            "selected_positions": baseline_positions,
            "keep_baseline": True,
            "edit_count": 0,
            "selection_steps": [],
        }

    pairs: List[Tuple[str, str]] = []
    for req in active_requirements:
        for pos in range(pool_limit):
            pairs.append((str(req.subquery), str(pool_docs[pos])))
    raw_scores = nli_score_fn(pairs)
    phi = np.zeros((len(active_requirements), pool_limit), dtype=float)
    idx = 0
    for r_idx in range(len(active_requirements)):
        for p_idx in range(pool_limit):
            phi[r_idx, p_idx] = float(raw_scores[idx])
            idx += 1

    finite_phi = phi[np.isfinite(phi)]
    tau = float(np.percentile(finite_phi, float(tau_percentile))) if finite_phi.size else 1.0
    tau = float(np.clip(tau, 0.0, 1.0))

    req_index_by_id = {str(req.unit_id): idx for idx, req in enumerate(active_requirements)}
    baseline_max_by_req = {
        str(req.unit_id): float(np.max(phi[idx, baseline_positions])) if baseline_positions else 0.0
        for idx, req in enumerate(active_requirements)
    }
    unmet_ids = {
        str(req_id)
        for req_id, score in baseline_max_by_req.items()
        if float(score) < tau
    }

    phi_stats = {
        "phi_mean": round(float(np.mean(phi)), 6),
        "phi_std": round(float(np.std(phi)), 6),
        "phi_min": round(float(np.min(phi)), 6),
        "phi_max": round(float(np.max(phi)), 6),
        "phi_median": round(float(np.median(phi)), 6),
        "phi_p25": round(float(np.percentile(phi, 25)), 6),
        "phi_p75": round(float(np.percentile(phi, 75)), 6),
    }

    if not unmet_ids:
        return baseline_positions, {
            "selection": selection_label,
            "status": "keep_all_demands_met",
            "query": str(query),
            "requirement_count": int(len(active_requirements)),
            "requirements": [req.to_trace() for req in active_requirements],
            "tau_percentile": round(float(tau_percentile), 4),
            "tau": round(float(tau), 6),
            "baseline_max_by_requirement": {
                req_id: round(float(score), 6) for req_id, score in baseline_max_by_req.items()
            },
            "unmet_requirement_ids": [],
            "selected_positions": baseline_positions,
            "selected_titles": [pool_doc_titles[pos] for pos in baseline_positions],
            "keep_baseline": True,
            "edit_count": 0,
            "edit_budget": int(max(0, edit_budget)),
            "selection_steps": [],
            "phi_stats": phi_stats,
        }

    def coverage_loss_count(out_pos: int, positions: Sequence[int]) -> int:
        selected = [int(pos) for pos in positions if 0 <= int(pos) < pool_limit]
        if int(out_pos) not in selected:
            return 0
        loss = 0
        for req_idx in range(len(active_requirements)):
            scores = [(pos, float(phi[req_idx, pos])) for pos in selected]
            scores.sort(key=lambda item: (-item[1], item[0]))
            if not scores or scores[0][0] != int(out_pos) or scores[0][1] < tau:
                continue
            second_score = scores[1][1] if len(scores) > 1 else 0.0
            if scores[0][1] > second_score + 1e-9:
                loss += 1
        return int(loss)

    selected_positions = list(baseline_positions)
    edit_limit = max(0, int(edit_budget))
    steps: List[Dict[str, object]] = []
    ordered_requirements = [
        req for req in _topological_requirement_order(active_requirements)
        if str(req.unit_id) in unmet_ids
    ]
    for req in ordered_requirements:
        if len(steps) >= edit_limit:
            break
        req_idx = req_index_by_id.get(str(req.unit_id))
        if req_idx is None:
            continue
        selected_set = set(selected_positions)
        candidates = [pos for pos in range(pool_limit) if pos not in selected_set]
        if not candidates:
            break
        in_pos = max(candidates, key=lambda pos: (float(phi[req_idx, pos]), -int(pos)))
        in_score = float(phi[req_idx, in_pos])
        if in_score < tau:
            continue
        out_pos = min(
            selected_positions,
            key=lambda pos: (coverage_loss_count(int(pos), selected_positions), int(pos)),
        )
        loss = coverage_loss_count(int(out_pos), selected_positions)
        if in_score <= float(loss):
            continue
        selected_positions = _replace_at_position(selected_positions, int(out_pos), int(in_pos))
        steps.append({
            "step": int(len(steps) + 1),
            "mode": "minimal_demand_repair_nli",
            "requirement_id": str(req.unit_id),
            "requirement_subquery": str(req.subquery),
            "out_position": int(out_pos),
            "out_title": str(pool_doc_titles[int(out_pos)]),
            "in_position": int(in_pos),
            "in_title": str(pool_doc_titles[int(in_pos)]),
            "in_phi": round(float(in_score), 6),
            "coverage_loss_count": int(loss),
            "tau": round(float(tau), 6),
        })

    keep_baseline = list(selected_positions) == baseline_positions
    return selected_positions, {
        "selection": selection_label,
        "status": "keep_no_accepted_repair" if keep_baseline else "repaired",
        "query": str(query),
        "requirement_count": int(len(active_requirements)),
        "requirements": [req.to_trace() for req in active_requirements],
        "tau_percentile": round(float(tau_percentile), 4),
        "tau": round(float(tau), 6),
        "baseline_max_by_requirement": {
            req_id: round(float(score), 6) for req_id, score in baseline_max_by_req.items()
        },
        "unmet_requirement_ids": sorted(unmet_ids),
        "selected_positions": list(selected_positions),
        "selected_titles": [pool_doc_titles[pos] for pos in selected_positions],
        "baseline_positions": baseline_positions,
        "baseline_titles": [pool_doc_titles[pos] for pos in baseline_positions],
        "keep_baseline": bool(keep_baseline),
        "edit_count": int(len(steps)),
        "edit_budget": int(edit_limit),
        "selection_steps": steps,
        "phi_stats": phi_stats,
    }


def select_coverage_aware_positions(
    *,
    query: str,
    requirements: Sequence[EvidenceNeed],
    requirement_embeddings: Dict[str, np.ndarray],
    pool_docs: Sequence[str],
    pool_doc_ids: Sequence[int | None],
    pool_doc_titles: Sequence[str],
    doc_idx_to_entities: Dict[int, set[str]],
    passage_embeddings: np.ndarray,
    qa_top_k: int,
    pool_doc_scores: Sequence[float] | None = None,
    binding_top_m: int = 5,
    embed_texts_fn: Callable[[Sequence[str]], Dict[str, np.ndarray]] | None = None,
    safe_projection: bool = False,
    safe_min_objective_gain: float = 0.02,
    safe_min_swap_gain: float = 0.01,
    safe_max_swaps: int = 2,
    safe_stability_window_m: int = 1,
    safe_projection_mode: str = "rank_cutoff",
    safe_agreement_top_k: int = 5,
    safe_retriever_margin_threshold: float = 1.01,
    safe_retriever_rank_penalty: float = 0.0,
    llm_extract_fn: Callable[[str, str], List[str]] | None = None,
    binding_mode: str = "auto",
    selective_binding_title_unique_threshold: float | None = None,
    llm_binding_title_match_mode: str = "substring",
    llm_binding_type_filter: bool = False,
    soft_compat_body_weight: float = 0.0,
    binding_grounding_enabled: bool = False,
    swap_refinement_enabled: bool = False,
    swap_min_gain: float = 0.001,
    gold_titles: Sequence[str] | None = None,
    evidence_link_metadata: Mapping[str, Any] | None = None,
    graph_prior_beta: float = 0.0,
    graph_prior_w_selected: float = 1.0,
    graph_prior_w_anchor: float = 0.3,
    graph_prior_w_rank: float = 0.2,
    selection_label_override: str | None = None,
) -> Tuple[List[int], Dict[str, object]]:
    """Select evidence by frozen-binding noisy-OR demand coverage.

    This is the clean coverage-aware selection composer path: it freezes an approximate binding set,
    precomputes phi[requirement, binding, document], and then runs greedy
    maximization over a single noisy-OR set objective. Historical gates, repair
    filters, and selection-time binding mutation intentionally do not
    participate in this path. The optional EvLink graph prior is additive and
    disabled by default, so beta=0 keeps the pure noisy-OR selection.
    """
    selection_label = selection_label_override or ("coverage_noisy_or_safe" if bool(safe_projection) else "coverage_noisy_or")
    pool_limit = len(pool_docs)
    target_k = min(max(int(qa_top_k), 0), pool_limit)
    if target_k <= 0:
        return [], {"selection": selection_label, "status": "empty_pool"}

    active_requirements = [
        req for req in requirements
        if str(req.subquery or "").strip() and not _is_operator_evidence_need(req)
    ]
    operator_requirement_count = len(list(requirements)) - len(active_requirements)
    if not active_requirements:
        fallback_positions = list(range(target_k))
        return fallback_positions, {
            "selection": selection_label,
            "status": "fallback_no_retrieval_requirements",
            "query": str(query),
            "requirement_count": 0,
            "operator_requirement_count": int(operator_requirement_count),
            "binding_count": 1,
            "selected_positions": fallback_positions,
            "selected_titles": [pool_doc_titles[pos] for pos in fallback_positions],
            "selection_steps": [],
        }

    doc_vectors: Dict[int, np.ndarray] = {}
    for pos, doc_id in enumerate(pool_doc_ids[:pool_limit]):
        if doc_id is None or int(doc_id) < 0 or int(doc_id) >= len(passage_embeddings):
            continue
        vector = _normalize_vector(passage_embeddings[int(doc_id)])
        if vector.size:
            doc_vectors[int(pos)] = vector

    raw_match_scores = _compute_embedding_match_scores(
        active_requirements,
        requirement_embeddings,
        doc_vectors,
        pool_limit,
    )
    req_by_id = {req.unit_id: req for req in active_requirements}
    normalized_query = normalize_structure_text(query)
    query_anchor_keys = {
        normalize_structure_text(anchor)
        for req in active_requirements
        for anchor in req.anchor_mentions
        if normalize_structure_text(anchor)
    }
    for title in pool_doc_titles[:pool_limit]:
        title_key = normalize_structure_text(title)
        if title_key and _contains_normalized_phrase(normalized_query, title_key):
            query_anchor_keys.add(title_key)

    def doc_entities_for_pos(pos: int) -> set[str]:
        doc_id = pool_doc_ids[pos] if 0 <= pos < len(pool_doc_ids) else None
        if doc_id is None:
            return set()
        return set(doc_idx_to_entities.get(int(doc_id), set()) or set())

    def collect_frozen_binding_candidates(req: EvidenceNeed) -> List[Dict[str, object]]:
        if not req.depends_on:
            return []
        rows: List[Dict[str, object]] = []
        seen_keys: set[str] = set()
        top_m = max(0, int(binding_top_m))
        for dep in req.depends_on:
            dep_req = req_by_id.get(dep)
            if dep_req is None:
                continue
            dep_scores = raw_match_scores.get(dep, np.zeros(pool_limit, dtype=float))
            upstream_positions = sorted(
                range(pool_limit),
                key=lambda pos: (-float(dep_scores[pos]), int(pos)),
            )[:top_m]
            expected_type = dep_req.expected_answer_type if dep_req is not None else "unknown"
            for dep_pos in upstream_positions:
                upstream_text = normalize_structure_text(_doc_text_body(pool_docs[dep_pos]))
                upstream_title_key = normalize_structure_text(pool_doc_titles[dep_pos])
                for title_pos, raw_title in enumerate(pool_doc_titles[:pool_limit]):
                    title = str(raw_title or "").strip()
                    title_key = normalize_structure_text(title)
                    if not title_key or title_key in seen_keys:
                        continue
                    if title_key == upstream_title_key or title_key in query_anchor_keys:
                        continue
                    if not _candidate_type_compatible(title, expected_type):
                        continue
                    count, first_pos = _phrase_occurrences(upstream_text, title_key)
                    if count <= 0:
                        continue
                    seen_keys.add(title_key)
                    rows.append({
                        "requirement_id": req.unit_id,
                        "title": title,
                        "key": title_key,
                        "dep": dep,
                        "dep_position": int(dep_pos),
                        "dep_score": float(dep_scores[dep_pos]),
                        "title_pool_position": int(title_pos),
                        "count": int(count),
                        "first_pos": int(first_pos),
                    })
        rows.sort(
            key=lambda row: (
                int(row["first_pos"]),
                -int(row["count"]),
                int(row["title_pool_position"]),
            )
        )
        return rows[:max(0, int(binding_top_m))]

    _llm_binding_title_match_mode = str(llm_binding_title_match_mode or "substring").strip().lower()
    allow_substring_title_match = _llm_binding_title_match_mode not in {"exact", "exact_only"}
    guarded_substring_title_match = _llm_binding_title_match_mode in {"substring_guarded", "guarded_substring"}
    wiki_title_match = _llm_binding_title_match_mode in {
        "wiki_title",
        "title_link",
        "normalized_title",
        "entity_title",
        "wiki_title_unique",
        "title_link_unique",
        "normalized_title_unique",
        "entity_title_unique",
    }
    require_unique_wiki_title_match = _llm_binding_title_match_mode in {
        "wiki_title_unique",
        "title_link_unique",
        "normalized_title_unique",
        "entity_title_unique",
    }
    llm_binding_extraction_traces: List[Dict[str, object]] = []

    def collect_llm_binding_candidates(req: EvidenceNeed) -> List[Dict[str, object]]:
        if not req.depends_on or llm_extract_fn is None:
            return []
        rows: List[Dict[str, object]] = []
        seen_keys: set[str] = set()
        top_m = max(0, int(binding_top_m))
        for dep in req.depends_on:
            dep_req = req_by_id.get(dep)
            if dep_req is None:
                continue
            dep_scores = raw_match_scores.get(dep, np.zeros(pool_limit, dtype=float))
            upstream_positions = sorted(
                range(pool_limit),
                key=lambda pos: (-float(dep_scores[pos]), int(pos)),
            )[:top_m]
            dep_subquery = str(dep_req.subquery or "").strip()
            if not dep_subquery:
                continue
            for dep_pos in upstream_positions:
                doc_text = str(pool_docs[dep_pos])
                entities = llm_extract_fn(dep_subquery, doc_text)
                extraction_trace: Dict[str, object] = {
                    "requirement_id": str(req.unit_id),
                    "dep": str(dep),
                    "dep_position": int(dep_pos),
                    "dep_title": str(pool_doc_titles[int(dep_pos)]),
                    "dep_score": round(float(dep_scores[dep_pos]), 6),
                    "dep_subquery": str(dep_subquery),
                    "expected_answer_type": str(dep_req.expected_answer_type if dep_req is not None else "unknown"),
                    "raw_entities": [str(ent) for ent in entities],
                    "matched_entities": [],
                    "unmatched_entities": [],
                    "skipped_anchor_entities": [],
                    "type_incompatible_entities": [],
                    "duplicate_entities": [],
                }
                for ent in entities:
                    ent_key = normalize_structure_text(ent)
                    if not ent_key:
                        continue
                    if ent_key in seen_keys:
                        extraction_trace["duplicate_entities"].append(str(ent))
                        continue
                    if ent_key in query_anchor_keys:
                        extraction_trace["skipped_anchor_entities"].append(str(ent))
                        continue
                    title_pos, match_type = _title_match_pool_position_and_type(
                        ent,
                        pool_doc_titles[:pool_limit],
                        allow_substring=allow_substring_title_match,
                        guarded_substring=guarded_substring_title_match,
                        wiki_title_match=wiki_title_match,
                        require_unique_wiki_title_match=require_unique_wiki_title_match,
                    )
                    if title_pos is None:
                        extraction_trace["unmatched_entities"].append(str(ent))
                        continue
                    matched_title = str(pool_doc_titles[title_pos])
                    if bool(llm_binding_type_filter) and not _candidate_type_compatible(
                        matched_title,
                        dep_req.expected_answer_type if dep_req is not None else "unknown",
                    ):
                        extraction_trace["type_incompatible_entities"].append({
                            "entity": str(ent),
                            "normalized_entity": str(ent_key),
                            "match_type": str(match_type),
                            "title": matched_title,
                            "title_pool_position": int(title_pos),
                            "expected_answer_type": str(dep_req.expected_answer_type if dep_req is not None else "unknown"),
                        })
                        continue
                    seen_keys.add(ent_key)
                    extraction_trace["matched_entities"].append({
                        "entity": str(ent),
                        "normalized_entity": str(ent_key),
                        "match_type": str(match_type),
                        "title": matched_title,
                        "title_pool_position": int(title_pos),
                    })
                    rows.append({
                        "requirement_id": req.unit_id,
                        "title": matched_title,
                        "key": ent_key,
                        "dep": dep,
                        "dep_position": int(dep_pos),
                        "dep_score": float(dep_scores[dep_pos]),
                        "title_pool_position": int(title_pos),
                        "count": 1,
                        "first_pos": 0,
                        "llm_extracted_entity": str(ent),
                        "entity_match_type": str(match_type),
                    })
                llm_binding_extraction_traces.append(extraction_trace)
        rows.sort(key=lambda row: (-float(row["dep_score"]), int(row["title_pool_position"])))
        return rows[:max(0, int(binding_top_m))]

    _binding_mode = str(binding_mode or "auto").strip().lower()
    if _binding_mode == "auto":
        _binding_mode = "llm" if llm_extract_fn is not None else "string_match"

    def collect_random_binding_candidates(req: EvidenceNeed) -> List[Dict[str, object]]:
        if not req.depends_on:
            return []
        import random as _rng
        eligible = [
            pos for pos in range(pool_limit)
            if normalize_structure_text(pool_doc_titles[pos]) not in query_anchor_keys
        ]
        if not eligible:
            return []
        chosen_pos = _rng.choice(eligible)
        return [{
            "requirement_id": req.unit_id,
            "title": str(pool_doc_titles[chosen_pos]),
            "key": normalize_structure_text(pool_doc_titles[chosen_pos]),
            "dep": str(req.depends_on[0]) if req.depends_on else "",
            "dep_position": 0,
            "dep_score": 0.0,
            "title_pool_position": int(chosen_pos),
            "count": 1,
            "first_pos": 0,
        }]

    def collect_oracle_binding_candidates(req: EvidenceNeed) -> List[Dict[str, object]]:
        if not req.depends_on or not gold_titles:
            return []
        rows: List[Dict[str, object]] = []
        seen_keys: set[str] = set()
        for gt in gold_titles:
            gt_key = normalize_structure_text(gt)
            if not gt_key or gt_key in seen_keys or gt_key in query_anchor_keys:
                continue
            title_pos = _title_match_pool_position(gt, pool_doc_titles[:pool_limit])
            if title_pos is None:
                continue
            seen_keys.add(gt_key)
            rows.append({
                "requirement_id": req.unit_id,
                "title": str(pool_doc_titles[title_pos]),
                "key": gt_key,
                "dep": str(req.depends_on[0]) if req.depends_on else "",
                "dep_position": 0,
                "dep_score": 1.0,
                "title_pool_position": int(title_pos),
                "count": 1,
                "first_pos": 0,
            })
        return rows[:max(0, int(binding_top_m))]

    use_llm_binding = _binding_mode in ("llm",)
    if _binding_mode == "nobind":
        binding_candidates_by_req = {}
    elif _binding_mode == "random":
        binding_candidates_by_req = {
            req.unit_id: collect_random_binding_candidates(req)
            for req in active_requirements
            if req.depends_on
        }
    elif _binding_mode == "oracle":
        binding_candidates_by_req = {
            req.unit_id: collect_oracle_binding_candidates(req)
            for req in active_requirements
            if req.depends_on
        }
    elif _binding_mode == "llm":
        binding_candidates_by_req = {
            req.unit_id: collect_llm_binding_candidates(req)
            for req in active_requirements
            if req.depends_on
        }
    else:
        binding_candidates_by_req = {
            req.unit_id: collect_frozen_binding_candidates(req)
            for req in active_requirements
            if req.depends_on
        }
    selective_binding_trace: Dict[str, object] = {
        "enabled": False,
        "policy": "disabled",
        "decision": "bind",
    }
    threshold_value: float | None = None
    if selective_binding_title_unique_threshold is not None:
        try:
            threshold_value = float(selective_binding_title_unique_threshold)
        except (TypeError, ValueError):
            threshold_value = None
        if threshold_value is not None and not np.isfinite(threshold_value):
            threshold_value = None
    if threshold_value is not None:
        stats = _binding_title_resolution_stats(binding_candidates_by_req, pool_doc_titles[:pool_limit])
        title_unique_rate = float(stats.get("title_unique_rate", 0.0) or 0.0)
        decision = "bind" if title_unique_rate >= float(threshold_value) else "abstain"
        selective_binding_trace = {
            "enabled": True,
            "policy": "query_level_title_unique",
            "threshold": round(float(threshold_value), 6),
            "score_name": "bind_conf_title_unique",
            "title_unique_rate": round(float(title_unique_rate), 6),
            "decision": decision,
            **stats,
        }
        if decision == "abstain":
            binding_candidates_by_req = {}
    candidate_lists: List[List[Dict[str, object]]] = [
        rows for rows in binding_candidates_by_req.values() if rows
    ]
    unpruned_binding_count = 1
    binding_pruned = False
    if candidate_lists:
        combo_records: List[Tuple[float, int, Tuple[Dict[str, object], ...]]] = []
        for combo_idx, combo in enumerate(itertools.product(*candidate_lists)):
            dep_score_sum = float(sum(float(row.get("dep_score", 0.0) or 0.0) for row in combo))
            combo_records.append((dep_score_sum, int(combo_idx), tuple(combo)))
        unpruned_binding_count = len(combo_records)
        if len(combo_records) > MAX_EVIDENCE_NEED_BINDINGS:
            binding_pruned = True
            combo_records = sorted(combo_records, key=lambda item: (-float(item[0]), int(item[1])))[:MAX_EVIDENCE_NEED_BINDINGS]
            combo_records = sorted(combo_records, key=lambda item: int(item[1]))

        bindings = []
        binding_assignment_rows_by_idx: List[Tuple[Dict[str, object], ...]] = []
        for binding_idx, (_, _, combo) in enumerate(combo_records):
            assignments = tuple(
                sorted(
                    (
                        str(row["requirement_id"]),
                        str(row["title"]),
                    )
                    for row in combo
                )
            )
            bindings.append(EvidenceNeedBinding(binding_id=f"b{binding_idx}", assignments=assignments))
            binding_assignment_rows_by_idx.append(tuple(dict(row) for row in combo))
    else:
        bindings = [EvidenceNeedBinding(binding_id="b0", assignments=())]
        binding_assignment_rows_by_idx = [tuple()]
    prior = 1.0 / float(max(len(bindings), 1))
    bindings = [
        EvidenceNeedBinding(binding_id=binding.binding_id, assignments=binding.assignments, prior=prior)
        for binding in bindings
    ]

    binding_embedding_cache: Dict[str, np.ndarray] = {}

    def score_bound_requirement(req: EvidenceNeed, candidate_title: str) -> np.ndarray:
        bound_query = _build_bound_subquery(req.subquery, candidate_title)
        if not bound_query or embed_texts_fn is None:
            return raw_match_scores.get(req.unit_id, np.zeros(pool_limit, dtype=float))
        if bound_query not in binding_embedding_cache:
            embedded = embed_texts_fn([bound_query]) or {}
            vector = embedded.get(bound_query) if isinstance(embedded, dict) else None
            binding_embedding_cache[bound_query] = _normalize_vector(vector if vector is not None else np.array([]))
        req_vec = binding_embedding_cache.get(bound_query, np.array([]))
        values: List[float] = []
        for pos in range(pool_limit):
            doc_vec = doc_vectors.get(pos)
            if req_vec.size == 0 or doc_vec is None or req_vec.shape != doc_vec.shape:
                values.append(float("-inf"))
            else:
                values.append(float(np.dot(req_vec, doc_vec)))
        return _min_max(np.asarray(values, dtype=float))

    phi = np.zeros((len(active_requirements), len(bindings), pool_limit), dtype=float)
    for req_idx, req in enumerate(active_requirements):
        for binding_idx, binding in enumerate(bindings):
            assignment = binding.assignment_map().get(req.unit_id, "")
            support_scores = raw_match_scores.get(req.unit_id, np.zeros(pool_limit, dtype=float))
            if assignment and req.depends_on:
                support_scores = score_bound_requirement(req, assignment)
                candidate_key = normalize_structure_text(assignment)
                if _binding_mode in ("llm", "random", "oracle"):
                    compat = np.asarray([
                        _llm_binding_compat_score(
                            assignment,
                            pool_doc_titles[pos],
                            pool_docs[pos],
                            body_mention_weight=float(soft_compat_body_weight),
                        )
                        for pos in range(pool_limit)
                    ], dtype=float)
                else:
                    compat = np.asarray([
                        _binding_candidate_hit_score(
                            candidate_key,
                            doc_entities_for_pos(pos),
                            pool_doc_titles[pos],
                            pool_docs[pos],
                        )
                        for pos in range(pool_limit)
                    ], dtype=float)
                support_scores = np.asarray(support_scores, dtype=float) * np.clip(compat, 0.0, 1.0)
            phi[req_idx, binding_idx, :] = np.clip(np.asarray(support_scores, dtype=float), 0.0, 1.0)

    demand_weights = np.full(len(active_requirements), 1.0 / float(max(len(active_requirements), 1)), dtype=float)
    graph_beta = max(float(graph_prior_beta), 0.0)
    graph_prior, graph_prior_trace = _build_evidence_link_graph_prior(
        evidence_link_metadata=evidence_link_metadata,
        pool_doc_ids=pool_doc_ids,
        pool_limit=pool_limit,
        w_selected=float(graph_prior_w_selected),
        w_anchor=float(graph_prior_w_anchor),
        w_rank=float(graph_prior_w_rank),
    )
    graph_prior_trace["beta"] = round(float(graph_beta), 6)
    graph_prior_trace["enabled"] = bool(
        graph_beta > 0.0
        and bool(evidence_link_metadata)
        and int((graph_prior_trace.get("stats") or {}).get("nonzero_count", 0) or 0) > 0
    )
    grounding_scores, grounding_traces = _binding_grounding_scores(
        phi=phi,
        active_requirements=active_requirements,
        bindings=bindings,
        binding_assignment_rows_by_idx=binding_assignment_rows_by_idx,
        pool_doc_titles=pool_doc_titles,
        pool_limit=pool_limit,
    )
    grounding_enabled = bool(binding_grounding_enabled)
    grounding_trace: Dict[str, object] = {
        "enabled": grounding_enabled,
        "protocol": "upstream_phi_entity_doc_mean",
        "scores": grounding_traces[:20],
    }

    def objective(binding_idx: int, positions: Sequence[int]) -> Tuple[float, np.ndarray]:
        coverage = _coverage_noisy_or(phi[:, binding_idx, :], positions)
        return float(np.dot(demand_weights, coverage)), coverage

    binding_results: List[Dict[str, object]] = []
    best_result: Dict[str, object] | None = None
    for binding_idx, binding in enumerate(bindings):
        selected_positions: List[int] = []
        selected_set: set[int] = set()
        current_score, current_coverage = objective(binding_idx, selected_positions)
        current_effective_score = float(current_score)
        current_graph_prior_score = 0.0
        steps: List[Dict[str, object]] = []
        while len(selected_positions) < target_k:
            best_row: Dict[str, object] | None = None
            for pos in range(pool_limit):
                if pos in selected_set:
                    continue
                proposed_positions = list(selected_positions) + [int(pos)]
                proposed_score, proposed_coverage = objective(binding_idx, proposed_positions)
                gain = float(proposed_score - current_score)
                graph_prior_value = float(graph_prior[int(pos)]) if int(pos) < graph_prior.size else 0.0
                graph_prior_gain = graph_beta * graph_prior_value
                effective_gain = gain + graph_prior_gain
                if graph_beta <= 0.0 and gain <= 1e-12:
                    continue
                if graph_beta > 0.0 and effective_gain <= 1e-12:
                    continue
                row = {
                    "pool_position": int(pos),
                    "title": pool_doc_titles[pos],
                    "objective_gain": gain,
                    "objective": float(proposed_score),
                    "graph_prior": graph_prior_value,
                    "graph_prior_gain": graph_prior_gain,
                    "effective_gain": effective_gain,
                    "effective_objective": current_effective_score + effective_gain,
                    "coverage": proposed_coverage,
                }
                if best_row is None or (
                    float(row["effective_gain"]),
                    float(row["objective_gain"]),
                    float(row["objective"]),
                    -int(row["pool_position"]),
                ) > (
                    float(best_row["effective_gain"]),
                    float(best_row["objective_gain"]),
                    float(best_row["objective"]),
                    -int(best_row["pool_position"]),
                ):
                    best_row = row
            if best_row is None:
                break
            chosen_pos = int(best_row["pool_position"])
            selected_positions.append(chosen_pos)
            selected_set.add(chosen_pos)
            current_score = float(best_row["objective"])
            current_coverage = np.asarray(best_row["coverage"], dtype=float)
            current_graph_prior_score += float(best_row["graph_prior_gain"])
            current_effective_score = float(best_row["effective_objective"])
            steps.append({
                "step": int(len(steps) + 1),
                "mode": "coverage_noisy_or_greedy",
                "pool_position": chosen_pos,
                "title": str(best_row["title"]),
                "objective_gain": round(float(best_row["objective_gain"]), 6),
                "objective": round(float(current_score), 6),
                "coverage_gain": round(float(best_row["objective_gain"]), 6),
                "graph_prior": round(float(best_row["graph_prior"]), 6),
                "graph_prior_gain": round(float(best_row["graph_prior_gain"]), 6),
                "effective_gain": round(float(best_row["effective_gain"]), 6),
                "effective_objective": round(float(current_effective_score), 6),
                "coverage_by_requirement": {
                    req.unit_id: round(float(current_coverage[req_idx]), 6)
                    for req_idx, req in enumerate(active_requirements)
                },
            })
        result = {
            "binding_idx": int(binding_idx),
            "binding": binding,
            "positions": selected_positions,
            "objective": float(current_score),
            "graph_prior_objective": float(current_graph_prior_score),
            "effective_objective": float(current_effective_score),
            "binding_grounding_score": float(grounding_scores[int(binding_idx)]) if int(binding_idx) < grounding_scores.size else 1.0,
            "binding_selection_score": (
                float(current_effective_score)
                * (
                    float(grounding_scores[int(binding_idx)])
                    if grounding_enabled and int(binding_idx) < grounding_scores.size
                    else 1.0
                )
            ),
            "coverage": current_coverage,
            "steps": steps,
        }
        binding_results.append(result)
        if best_result is None or (
            float(result["binding_selection_score"]),
            float(result["effective_objective"]),
            float(result["objective"]),
            -int(result["binding_idx"]),
        ) > (
            float(best_result["binding_selection_score"]),
            float(best_result["effective_objective"]),
            float(best_result["objective"]),
            -int(best_result["binding_idx"]),
        ):
            best_result = result

    if best_result is None:
        fallback_positions = list(range(target_k))
        return fallback_positions, {
            "selection": selection_label,
            "status": "fallback_no_positive_objective",
            "query": str(query),
            "requirement_count": int(len(active_requirements)),
            "operator_requirement_count": int(operator_requirement_count),
            "binding_count": int(len(bindings)),
            "binding_count_unpruned": int(unpruned_binding_count),
            "binding_max_bindings": int(MAX_EVIDENCE_NEED_BINDINGS),
            "binding_pruned": bool(binding_pruned),
            "phi_shape": [int(dim) for dim in phi.shape],
            "evidence_link_graph_prior": graph_prior_trace,
            "binding_grounding": grounding_trace,
            "selected_positions": fallback_positions,
            "selected_titles": [pool_doc_titles[pos] for pos in fallback_positions],
            "selection_steps": [],
        }
    best_binding = best_result["binding"]
    best_binding_idx = int(best_result["binding_idx"])
    rebuild_positions = list(best_result["positions"])[:target_k]
    rebuild_objective = float(best_result["objective"])
    rebuild_effective_objective = float(best_result.get("effective_objective", rebuild_objective))
    rebuild_coverage = np.asarray(best_result["coverage"], dtype=float)
    baseline_positions = list(range(target_k))
    baseline_objective, baseline_coverage = objective(best_binding_idx, baseline_positions)
    rank_scores = _coverage_rank_scores(pool_doc_scores, pool_limit)
    retriever_margin = 0.0
    if rank_scores.size and target_k > 0:
        retriever_margin = float(rank_scores[0] - rank_scores[target_k - 1])

    projection_mode = str(safe_projection_mode or "rank_cutoff").strip().lower()
    safe_trace: Dict[str, object] = {
        "safe_projection": bool(safe_projection),
        "safe_projection_mode": projection_mode,
        "baseline_positions": list(baseline_positions),
        "baseline_titles": [pool_doc_titles[pos] for pos in baseline_positions],
        "baseline_objective": round(float(baseline_objective), 6),
        "rebuild_positions": list(rebuild_positions),
        "rebuild_titles": [pool_doc_titles[pos] for pos in rebuild_positions],
        "rebuild_objective": round(float(rebuild_objective), 6),
        "rebuild_effective_objective": round(float(rebuild_effective_objective), 6),
        "rebuild_gain_over_baseline": round(float(rebuild_objective - baseline_objective), 6),
        "retriever_margin_top1_to_topk": round(float(retriever_margin), 6),
        "safe_min_objective_gain": round(float(safe_min_objective_gain), 6),
        "safe_min_swap_gain": round(float(safe_min_swap_gain), 6),
        "safe_max_swaps": int(max(0, safe_max_swaps)),
        "safe_stability_window_m": int(max(0, safe_stability_window_m)),
        "safe_agreement_top_k": int(max(1, safe_agreement_top_k)),
        "safe_retriever_margin_threshold": round(float(safe_retriever_margin_threshold), 6),
        "safe_retriever_rank_penalty": round(float(safe_retriever_rank_penalty), 6),
        "safe_decision": "not_enabled",
        "safe_swap_steps": [],
    }

    best_positions = rebuild_positions
    best_coverage = rebuild_coverage
    best_objective = rebuild_objective
    best_steps = list(best_result["steps"])
    swap_refinement_trace: Dict[str, object] = {
        "enabled": bool(swap_refinement_enabled),
        "min_gain": round(float(swap_min_gain), 6),
        "applied": False,
        "steps": [],
    }
    if bool(safe_projection):
        max_swaps = max(0, int(safe_max_swaps))
        stability_window_m = max(0, int(safe_stability_window_m))
        min_total_gain = float(safe_min_objective_gain)
        min_swap_gain = float(safe_min_swap_gain)
        rank_penalty = max(0.0, float(safe_retriever_rank_penalty))
        agreement_projection = projection_mode in {
            "agreement_r2_ge",
            "agreement_r2_gt",
            "agreement_r3_ge",
            "agreement_r3_gt",
        }
        if (
            np.isfinite(float(safe_retriever_margin_threshold))
            and float(safe_retriever_margin_threshold) <= 1.0
            and retriever_margin >= float(safe_retriever_margin_threshold)
        ):
            best_positions = list(baseline_positions)
            best_objective = float(baseline_objective)
            best_coverage = np.asarray(baseline_coverage, dtype=float)
            best_steps = []
            safe_trace["safe_decision"] = "fallback_retriever_margin"
        elif (not agreement_projection) and float(rebuild_objective - baseline_objective) < min_total_gain:
            best_positions = list(baseline_positions)
            best_objective = float(baseline_objective)
            best_coverage = np.asarray(baseline_coverage, dtype=float)
            best_steps = []
            safe_trace["safe_decision"] = "fallback_low_rebuild_gain"
        else:
            current_positions = list(baseline_positions)
            current_score = float(baseline_objective)
            current_coverage = np.asarray(baseline_coverage, dtype=float)
            swap_steps: List[Dict[str, object]] = []
            if agreement_projection:
                include_coverage_signal = "_r3_" in projection_mode
                strict_retention = projection_mode.endswith("_gt")
                retention_scores, retention_trace = _agreement_retention_scores(
                    pool_doc_ids=pool_doc_ids,
                    pool_limit=pool_limit,
                    baseline_positions=baseline_positions,
                    rebuild_positions=rebuild_positions,
                    evidence_link_metadata=evidence_link_metadata,
                    include_coverage_signal=include_coverage_signal,
                    agreement_top_k=int(safe_agreement_top_k),
                )
                safe_trace["agreement_retention"] = retention_trace
                safe_trace["agreement_retention_rule"] = "gt" if strict_retention else "ge"
                for _ in range(target_k):
                    current_set = set(current_positions)
                    best_swap: Dict[str, object] | None = None
                    for out_pos in list(current_positions):
                        out_retention = float(retention_scores[int(out_pos)]) if int(out_pos) < retention_scores.size else 0.0
                        for in_pos in range(pool_limit):
                            if in_pos in current_set:
                                continue
                            in_retention = float(retention_scores[int(in_pos)]) if int(in_pos) < retention_scores.size else 0.0
                            if strict_retention:
                                retention_allowed = in_retention > out_retention
                            else:
                                retention_allowed = in_retention >= out_retention
                            if not retention_allowed:
                                continue
                            proposed_positions = _replace_at_position(current_positions, out_pos, in_pos)
                            proposed_score, proposed_coverage = objective(best_binding_idx, proposed_positions)
                            gain = float(proposed_score - current_score)
                            if gain <= min_swap_gain:
                                continue
                            row = {
                                "out_position": int(out_pos),
                                "out_title": pool_doc_titles[int(out_pos)],
                                "out_retention": out_retention,
                                "in_position": int(in_pos),
                                "in_title": pool_doc_titles[int(in_pos)],
                                "in_retention": in_retention,
                                "objective_gain": gain,
                                "objective": float(proposed_score),
                                "coverage": proposed_coverage,
                            }
                            if best_swap is None or (
                                float(row["objective_gain"]),
                                float(row["in_retention"] - row["out_retention"]),
                                float(row["objective"]),
                                -int(row["in_position"]),
                            ) > (
                                float(best_swap["objective_gain"]),
                                float(best_swap["in_retention"] - best_swap["out_retention"]),
                                float(best_swap["objective"]),
                                -int(best_swap["in_position"]),
                            ):
                                best_swap = row
                    if best_swap is None:
                        break
                    current_positions = _replace_at_position(
                        current_positions,
                        int(best_swap["out_position"]),
                        int(best_swap["in_position"]),
                    )
                    current_score = float(best_swap["objective"])
                    current_coverage = np.asarray(best_swap["coverage"], dtype=float)
                    swap_steps.append({
                        "step": int(len(swap_steps) + 1),
                        "mode": "coverage_noisy_or_agreement_admission",
                        "out_position": int(best_swap["out_position"]),
                        "out_title": str(best_swap["out_title"]),
                        "out_retention": round(float(best_swap["out_retention"]), 6),
                        "in_position": int(best_swap["in_position"]),
                        "in_title": str(best_swap["in_title"]),
                        "in_retention": round(float(best_swap["in_retention"]), 6),
                        "objective_gain": round(float(best_swap["objective_gain"]), 6),
                        "objective": round(float(current_score), 6),
                        "coverage_by_requirement": {
                            req.unit_id: round(float(current_coverage[req_idx]), 6)
                            for req_idx, req in enumerate(active_requirements)
                        },
                    })
            else:
                stability_guard_positions = set(range(min(stability_window_m, target_k)))
                for _ in range(max_swaps):
                    current_set = set(current_positions)
                    best_swap: Dict[str, object] | None = None
                    for out_pos in list(current_positions):
                        if int(out_pos) in stability_guard_positions:
                            continue
                        for in_pos in range(pool_limit):
                            if in_pos in current_set:
                                continue
                            proposed_positions = _replace_at_position(current_positions, out_pos, in_pos)
                            proposed_score, proposed_coverage = objective(best_binding_idx, proposed_positions)
                            gain = float(proposed_score - current_score)
                            retriever_rank_loss = 0.0
                            if rank_scores.size:
                                retriever_rank_loss = max(
                                    0.0,
                                    float(rank_scores[int(out_pos)]) - float(rank_scores[int(in_pos)]),
                                )
                            adjusted_gain = gain - rank_penalty * retriever_rank_loss
                            if adjusted_gain < min_swap_gain:
                                continue
                            row = {
                                "out_position": int(out_pos),
                                "out_title": pool_doc_titles[int(out_pos)],
                                "in_position": int(in_pos),
                                "in_title": pool_doc_titles[int(in_pos)],
                                "objective_gain": gain,
                                "retriever_rank_loss": retriever_rank_loss,
                                "adjusted_gain": adjusted_gain,
                                "objective": float(proposed_score),
                                "coverage": proposed_coverage,
                            }
                            if best_swap is None or (
                                float(row["adjusted_gain"]),
                                float(row["objective_gain"]),
                                float(row["objective"]),
                                -int(row["in_position"]),
                            ) > (
                                float(best_swap["adjusted_gain"]),
                                float(best_swap["objective_gain"]),
                                float(best_swap["objective"]),
                                -int(best_swap["in_position"]),
                            ):
                                best_swap = row
                    if best_swap is None:
                        break
                    current_positions = _replace_at_position(
                        current_positions,
                        int(best_swap["out_position"]),
                        int(best_swap["in_position"]),
                    )
                    current_score = float(best_swap["objective"])
                    current_coverage = np.asarray(best_swap["coverage"], dtype=float)
                    swap_steps.append({
                        "step": int(len(swap_steps) + 1),
                        "mode": "coverage_noisy_or_safe_admission",
                        "out_position": int(best_swap["out_position"]),
                        "out_title": str(best_swap["out_title"]),
                        "in_position": int(best_swap["in_position"]),
                        "in_title": str(best_swap["in_title"]),
                        "objective_gain": round(float(best_swap["objective_gain"]), 6),
                        "retriever_rank_loss": round(float(best_swap["retriever_rank_loss"]), 6),
                        "adjusted_gain": round(float(best_swap["adjusted_gain"]), 6),
                        "objective": round(float(current_score), 6),
                        "coverage_by_requirement": {
                            req.unit_id: round(float(current_coverage[req_idx]), 6)
                            for req_idx, req in enumerate(active_requirements)
                        },
                    })
            best_positions = list(current_positions)
            best_objective = float(current_score)
            best_coverage = np.asarray(current_coverage, dtype=float)
            best_steps = list(swap_steps)
            safe_trace["safe_swap_steps"] = list(swap_steps)
            if agreement_projection:
                safe_trace["safe_decision"] = "agreement_edit_applied" if swap_steps else "fallback_no_agreement_swap"
            else:
                safe_trace["safe_decision"] = "minimal_edit_applied" if swap_steps else "fallback_no_eligible_swap"
    elif bool(swap_refinement_enabled):
        refined_positions, refined_objective, refined_coverage, refinement_steps = _refine_by_single_swaps(
            binding_idx=best_binding_idx,
            initial_positions=best_positions,
            pool_limit=pool_limit,
            objective_fn=objective,
            min_gain=float(swap_min_gain),
            max_passes=1,
        )
        if refinement_steps:
            best_positions = list(refined_positions)
            best_objective = float(refined_objective)
            best_coverage = np.asarray(refined_coverage, dtype=float)
            best_steps = list(best_steps) + list(refinement_steps)
            swap_refinement_trace["applied"] = True
            swap_refinement_trace["steps"] = list(refinement_steps)

    covered_count = int(np.sum(best_coverage > 0.0))
    trace = {
        "selection": selection_label,
        "status": "applied",
        "query": str(query),
        "requirement_count": int(len(active_requirements)),
        "operator_requirement_count": int(operator_requirement_count),
        "requirements": [req.to_trace() for req in active_requirements],
        "binding_top_m": int(binding_top_m),
        "binding_count": int(len(bindings)),
        "binding_count_unpruned": int(unpruned_binding_count),
        "binding_max_bindings": int(MAX_EVIDENCE_NEED_BINDINGS),
        "binding_pruned": bool(binding_pruned),
        "binding_selection_protocol": "best_binding_final_pool",
        "binding_mode": str(_binding_mode),
        "effective_binding_mode": "nobind" if selective_binding_trace.get("decision") == "abstain" else str(_binding_mode),
        "selective_binding": selective_binding_trace,
        "selective_binding_enabled": bool(selective_binding_trace.get("enabled", False)),
        "selective_binding_decision": str(selective_binding_trace.get("decision", "bind")),
        "selective_binding_title_unique_rate": float(selective_binding_trace.get("title_unique_rate", 0.0) or 0.0),
        "selective_binding_threshold": selective_binding_trace.get("threshold"),
        "llm_binding_title_match_mode": str(_llm_binding_title_match_mode),
        "llm_binding_type_filter": bool(llm_binding_type_filter),
        "soft_compat_body_weight": round(float(soft_compat_body_weight), 6),
        "llm_binding_extractions": llm_binding_extraction_traces[:50],
        "llm_binding_extraction_count": int(len(llm_binding_extraction_traces)),
        "bindings": [binding.to_trace() for binding in bindings[:20]],
        "selected_binding": best_binding.to_trace(),
        "selected_binding_id": str(best_binding.binding_id),
        "selected_binding_grounding_score": round(
            float(grounding_scores[best_binding_idx]) if best_binding_idx < grounding_scores.size else 1.0,
            6,
        ),
        "selected_binding_score": round(float(best_result.get("binding_selection_score", rebuild_effective_objective)), 6),
        "binding_grounding": grounding_trace,
        "binding_candidates_by_requirement": {
            req_id: [
                {
                    "title": str(row.get("title", "") or ""),
                    "dep": str(row.get("dep", "") or ""),
                    "dep_position": int(row.get("dep_position", -1)),
                    "dep_score": round(float(row.get("dep_score", 0.0) or 0.0), 6),
                    "title_pool_position": int(row.get("title_pool_position", -1)),
                    "count": int(row.get("count", 0)),
                    "first_pos": int(row.get("first_pos", 10**9)),
                    "llm_extracted_entity": str(row.get("llm_extracted_entity", "") or ""),
                    "entity_match_type": str(row.get("entity_match_type", "") or ""),
                }
                for row in rows
            ]
            for req_id, rows in binding_candidates_by_req.items()
        },
        "support_matrix_schema_version": SUPPORT_MATRIX_SCHEMA_VERSION,
        "support_matrix": _support_matrix_rows(
            active_requirements=active_requirements,
            phi=phi,
            binding_idx=best_binding_idx,
            binding=best_binding,
            pool_doc_titles=pool_doc_titles,
            baseline_positions=baseline_positions,
            final_positions=best_positions,
        ),
        "objective": round(float(best_objective), 6),
        "rebuild_objective": round(float(rebuild_objective), 6),
        "baseline_objective": round(float(baseline_objective), 6),
        "covered_requirement_count": covered_count,
        "covered_requirement_rate": round(float(covered_count) / float(max(len(active_requirements), 1)), 4),
        "coverage_by_requirement": {
            req.unit_id: round(float(best_coverage[req_idx]), 6)
            for req_idx, req in enumerate(active_requirements)
        },
        "phi_shape": [int(dim) for dim in phi.shape],
        "selected_positions": best_positions,
        "selected_titles": [pool_doc_titles[pos] for pos in best_positions],
        "selection_steps": list(best_steps),
        "evidence_link_graph_prior": graph_prior_trace,
        "swap_refinement_trace": swap_refinement_trace,
        "safe_projection_trace": safe_trace,
        "embedding_available_requirement_count": int(
            sum(_normalize_vector(requirement_embeddings.get(req.unit_id, np.array([]))).size > 0 for req in active_requirements)
        ),
        "embedding_available_doc_count": int(len(doc_vectors)),
        "binding_objectives": [
            {
                "binding_id": str(result["binding"].binding_id),
                "objective": round(float(result["objective"]), 6),
                "graph_prior_objective": round(float(result.get("graph_prior_objective", 0.0)), 6),
                "effective_objective": round(float(result.get("effective_objective", result["objective"])), 6),
                "binding_grounding_score": round(float(result.get("binding_grounding_score", 1.0)), 6),
                "binding_selection_score": round(float(result.get("binding_selection_score", result.get("effective_objective", result["objective"]))), 6),
                "selected_positions": list(result["positions"]),
            }
            for result in binding_results[:20]
        ],
    }
    return best_positions, trace


def select_evidence_need_embed_positions(
    *,
    query: str,
    requirements: Sequence[EvidenceNeed],
    requirement_embeddings: Dict[str, np.ndarray],
    pool_docs: Sequence[str],
    pool_doc_ids: Sequence[int | None],
    pool_doc_scores: Sequence[float],
    pool_doc_titles: Sequence[str],
    doc_idx_to_entities: Dict[int, set[str]],
    passage_embeddings: np.ndarray,
    qa_top_k: int,
    reserve_top_m: int = 2,
    match_threshold: float = 0.35,
    redundancy_weight: float = 0.10,
    base_weight: float = 0.05,
    rank_weight: float = 0.0,
    anchor_bonus_weight: float = 0.10,
    dependency_bonus_weight: float = 0.10,
    non_anchor_title_dedup: bool = True,
    enable_dependency_binding: bool = False,
    enforce_dependencies: bool = True,
    binding_max_candidates: int = 4,
    binding_entity_hit_required: bool = True,
    require_new_crossing: bool = False,
    demand_gate_enabled: bool = False,
    demand_gate_alpha: float = 1.0,
    repairable_filter_enabled: bool = False,
    satisfiable_by_policy: str = "binding_override",
    ser_enabled: bool = False,
    ser_lambda0: float = 1.0,
    ser_anchor_binding_enabled: bool = False,
    ser_repairable_gap_enabled: bool = False,
    embed_texts_fn: Callable[[Sequence[str]], Dict[str, np.ndarray]] | None = None,
) -> Tuple[List[int], Dict[str, object]]:
    pool_limit = len(pool_docs)
    normalized_satisfiable_by_policy = _normalize_satisfiable_by_policy(satisfiable_by_policy)
    target_k = min(max(int(qa_top_k), 0), pool_limit)
    if target_k <= 0:
        return [], {"selection": "evidence_need_embed", "status": "empty_pool"}

    active_requirements = list(requirements)
    if not enforce_dependencies:
        active_requirements = [
            EvidenceNeed(
                unit_id=req.unit_id,
                subquery=req.subquery,
                depends_on=(),
                expected_answer_type=req.expected_answer_type,
                anchor_mentions=req.anchor_mentions,
                role=req.role,
                satisfiable_by=req.satisfiable_by,
            )
            for req in active_requirements
        ]
    if not active_requirements:
        fallback_positions = list(range(target_k))
        return fallback_positions, {
            "selection": "evidence_need_embed",
            "status": "fallback_no_requirements",
            "selection_steps": [],
            "final_positions": fallback_positions,
        }

    doc_vectors: Dict[int, np.ndarray] = {}
    for pos, doc_id in enumerate(pool_doc_ids[:pool_limit]):
        if doc_id is None or int(doc_id) < 0 or int(doc_id) >= len(passage_embeddings):
            continue
        vector = _normalize_vector(passage_embeddings[int(doc_id)])
        if vector.size:
            doc_vectors[int(pos)] = vector

    req_vectors = {
        req.unit_id: _normalize_vector(requirement_embeddings.get(req.unit_id, np.array([])))
        for req in active_requirements
    }
    raw_match_scores: Dict[str, np.ndarray] = {}
    for req in active_requirements:
        req_vec = req_vectors.get(req.unit_id, np.array([]))
        values: List[float] = []
        for pos in range(pool_limit):
            doc_vec = doc_vectors.get(pos)
            if req_vec.size == 0 or doc_vec is None or req_vec.shape != doc_vec.shape:
                values.append(float("-inf"))
            else:
                values.append(float(np.dot(req_vec, doc_vec)))
        raw_match_scores[req.unit_id] = _min_max(np.asarray(values, dtype=float))

    normalized_base_scores = _min_max(np.asarray(pool_doc_scores[:pool_limit], dtype=float))
    normalized_ranks = (
        np.asarray([float(pos) / float(max(pool_limit - 1, 1)) for pos in range(pool_limit)], dtype=float)
        if pool_limit > 0
        else np.asarray([], dtype=float)
    )
    normalized_query = normalize_structure_text(query)
    query_anchor_keys = {
        normalize_structure_text(anchor)
        for req in active_requirements
        for anchor in req.anchor_mentions
        if normalize_structure_text(anchor)
    }
    for title in pool_doc_titles[:pool_limit]:
        title_key = normalize_structure_text(title)
        if title_key and _contains_normalized_phrase(normalized_query, title_key):
            query_anchor_keys.add(title_key)

    selected_positions: List[int] = []
    seen_titles: set[str] = set()
    reserve_count = min(max(int(reserve_top_m), 0), target_k, pool_limit)
    for pos in range(reserve_count):
        selected_positions.append(pos)
        title_key = normalize_structure_text(pool_doc_titles[pos]) if pos < len(pool_doc_titles) else ""
        if title_key:
            seen_titles.add(title_key)

    req_by_id = {req.unit_id: req for req in active_requirements}
    coverage_by_req = {req.unit_id: 0.0 for req in active_requirements}
    cover_position_by_req: Dict[str, int] = {}
    dependency_entities_by_req: Dict[str, set[str]] = {}
    binding_candidates_by_req: Dict[str, List[Dict[str, object]]] = {}
    binding_embedding_cache: Dict[str, np.ndarray] = {}
    binding_score_cache: Dict[Tuple[str, str], np.ndarray] = {}
    selection_steps: List[Dict[str, object]] = []

    def doc_entities_for_pos(pos: int) -> set[str]:
        doc_id = pool_doc_ids[pos] if 0 <= pos < len(pool_doc_ids) else None
        if doc_id is None:
            return set()
        return set(doc_idx_to_entities.get(int(doc_id), set()) or set())

    def collect_binding_candidates(
        req: EvidenceNeed,
        dep_entities: set[str],
        cover_positions: Dict[str, int] | None = None,
    ) -> List[Dict[str, object]]:
        if not enable_dependency_binding or not req.depends_on:
            return []
        resolved_cover_positions = cover_position_by_req if cover_positions is None else cover_positions
        rows: List[Dict[str, object]] = []
        seen_keys: set[str] = set()
        for dep in req.depends_on:
            dep_pos = resolved_cover_positions.get(dep)
            if dep_pos is None or dep_pos < 0 or dep_pos >= pool_limit:
                continue
            dep_req = req_by_id.get(dep)
            expected_type = dep_req.expected_answer_type if dep_req is not None else "unknown"
            upstream_text = normalize_structure_text(_doc_text_body(pool_docs[dep_pos]))
            upstream_title_key = normalize_structure_text(pool_doc_titles[dep_pos])
            for title_pos, raw_title in enumerate(pool_doc_titles[:pool_limit]):
                title = str(raw_title or "").strip()
                title_key = normalize_structure_text(title)
                if not title_key or title_key in seen_keys:
                    continue
                if title_key == upstream_title_key or title_key in query_anchor_keys:
                    continue
                if not _candidate_type_compatible(title, expected_type):
                    continue
                count, first_pos = _phrase_occurrences(upstream_text, title_key)
                if count <= 0:
                    continue
                seen_keys.add(title_key)
                rows.append({
                    "title": title,
                    "key": title_key,
                    "dep": dep,
                    "dep_position": int(dep_pos),
                    "title_pool_position": int(title_pos),
                    "count": int(count),
                    "first_pos": int(first_pos),
                    "expected_answer_type": str(expected_type or "unknown"),
                })
        rows.sort(key=lambda row: (int(row["first_pos"]), -int(row["count"]), int(row["title_pool_position"])))
        return rows[:max(0, int(binding_max_candidates))]

    def ensure_bound_score(req: EvidenceNeed, candidate: Dict[str, object]) -> np.ndarray | None:
        candidate_key = str(candidate.get("key", "") or "")
        if not candidate_key:
            return None
        cache_key = (req.unit_id, candidate_key)
        cached_score = binding_score_cache.get(cache_key)
        if cached_score is not None:
            return cached_score
        candidate_title = str(candidate.get("title", "") or "")
        bound_query = _build_bound_subquery(req.subquery, candidate_title)
        if not bound_query:
            return None
        if bound_query not in binding_embedding_cache:
            if embed_texts_fn is None:
                return None
            embedded = embed_texts_fn([bound_query]) or {}
            vector = embedded.get(bound_query) if isinstance(embedded, dict) else None
            if vector is None:
                return None
            binding_embedding_cache[bound_query] = _normalize_vector(vector)
        req_vec = binding_embedding_cache.get(bound_query, np.array([]))
        values: List[float] = []
        for pos in range(pool_limit):
            doc_vec = doc_vectors.get(pos)
            if req_vec.size == 0 or doc_vec is None or req_vec.shape != doc_vec.shape:
                values.append(float("-inf"))
            else:
                values.append(float(np.dot(req_vec, doc_vec)))
        scores = _min_max(np.asarray(values, dtype=float))
        binding_score_cache[cache_key] = scores
        return scores

    def score_requirement_for_position(
        req: EvidenceNeed,
        pos: int,
        *,
        dependency_entities: set[str] | None = None,
        binding_candidates_override: List[Dict[str, object]] | None = None,
    ) -> Tuple[float, Dict[str, object]]:
        dep_entities = (
            dependency_entities
            if dependency_entities is not None
            else dependency_entities_by_req.get(req.unit_id, set())
        )
        req_score = float(raw_match_scores.get(req.unit_id, np.zeros(pool_limit))[pos])
        anchor_score = _anchor_hit_score(
            req.anchor_mentions,
            pool_doc_titles[pos],
            pool_docs[pos],
            doc_entities_for_pos(pos),
        )
        dep_score = _dependency_hit_score(
            dep_entities,
            doc_entities_for_pos(pos),
            pool_doc_titles[pos],
        )
        binding_candidates = (
            binding_candidates_override
            if binding_candidates_override is not None
            else binding_candidates_by_req.get(req.unit_id, [])
        )
        binding_rows: List[Dict[str, object]] = []
        best_binding_score = 0.0
        best_binding_hit = 0.0
        best_binding_title = ""
        if binding_candidates:
            for candidate in binding_candidates:
                candidate_key = str(candidate.get("key", "") or "")
                hit_score = _binding_candidate_hit_score(
                    candidate_key,
                    doc_entities_for_pos(pos),
                    pool_doc_titles[pos],
                    pool_docs[pos],
                )
                bound_scores = ensure_bound_score(req, candidate)
                bound_score = float(bound_scores[pos]) if bound_scores is not None else 0.0
                if binding_entity_hit_required and hit_score <= 0.0:
                    effective = 0.0
                else:
                    effective = min(1.0, max(0.0, bound_score + float(dependency_bonus_weight) * hit_score))
                binding_rows.append({
                    "title": str(candidate.get("title", "") or ""),
                    "hit": round(float(hit_score), 4),
                    "score": round(float(bound_score), 4),
                    "effective": round(float(effective), 4),
                })
                if effective > best_binding_score:
                    best_binding_score = float(effective)
                    best_binding_hit = float(hit_score)
                    best_binding_title = str(candidate.get("title", "") or "")
            req_score = best_binding_score
            dep_score = max(dep_score, best_binding_hit)
        elif req.depends_on and dep_entities and dep_score <= 0.0:
            req_score = 0.0
        else:
            req_score = min(
                1.0,
                max(
                    0.0,
                    req_score
                    + float(anchor_bonus_weight) * anchor_score
                    + float(dependency_bonus_weight) * dep_score,
                ),
            )
        return min(1.0, max(0.0, req_score)), {
            "anchor_score": round(float(anchor_score), 4),
            "dependency_score": round(float(dep_score), 4),
            "binding_title": best_binding_title,
            "binding_candidates": binding_rows,
        }

    def mark_coverage_from_position(pos: int, *, mode: str) -> None:
        made_progress = True
        while made_progress:
            made_progress = False
            selected_entity_context = {
                normalize_structure_text(entity)
                for selected_pos in selected_positions
                for entity in doc_entities_for_pos(selected_pos)
            }
            for selected_pos in selected_positions:
                title_key = normalize_structure_text(pool_doc_titles[selected_pos])
                if title_key:
                    selected_entity_context.add(title_key)
            for req in active_requirements:
                if any(dep not in cover_position_by_req for dep in req.depends_on):
                    continue
                dep_entities: set[str] = set()
                for dep in req.depends_on:
                    dep_pos = cover_position_by_req.get(dep)
                    if dep_pos is None:
                        continue
                    dep_entities.update(
                        normalize_structure_text(entity)
                        for entity in doc_entities_for_pos(dep_pos)
                    )
                    dep_title = normalize_structure_text(pool_doc_titles[dep_pos])
                    if dep_title:
                        dep_entities.add(dep_title)
                dependency_entities_by_req[req.unit_id] = dep_entities
                binding_candidates_by_req[req.unit_id] = collect_binding_candidates(req, dep_entities)
                score, score_trace = score_requirement_for_position(req, pos)
                previous_score = float(coverage_by_req.get(req.unit_id, 0.0))
                previous_cover_position = cover_position_by_req.get(req.unit_id)
                if score > previous_score:
                    coverage_by_req[req.unit_id] = score
                    should_set_cover = score >= float(match_threshold) and (
                        req.unit_id not in cover_position_by_req or bool(enable_dependency_binding)
                    )
                    if should_set_cover:
                        cover_position_by_req[req.unit_id] = int(pos)
                        if previous_cover_position != int(pos):
                            made_progress = True
                        selection_steps.append({
                            "step": int(len(selection_steps) + 1),
                            "mode": mode,
                            "covered_requirement_id": req.unit_id,
                            "cover_update": previous_cover_position is not None,
                            "pool_position": int(pos),
                            "title": pool_doc_titles[pos],
                            "coverage_score": round(float(score), 4),
                            "previous_coverage_score": round(float(previous_score), 4),
                            "binding_title": score_trace.get("binding_title", ""),
                        })

    def compute_demand_assessment(positions: Sequence[int]) -> Dict[str, object]:
        local_coverage_by_req = {req.unit_id: 0.0 for req in active_requirements}
        local_cover_position_by_req: Dict[str, int] = {}
        local_binding_candidates_by_req: Dict[str, List[Dict[str, object]]] = {}
        local_steps: List[Dict[str, object]] = []
        valid_positions = [
            int(pos)
            for pos in positions
            if 0 <= int(pos) < pool_limit
        ]
        for pos in valid_positions:
            made_progress = True
            while made_progress:
                made_progress = False
                for req in active_requirements:
                    if any(dep not in local_cover_position_by_req for dep in req.depends_on):
                        continue
                    dep_entities: set[str] = set()
                    for dep in req.depends_on:
                        dep_pos = local_cover_position_by_req.get(dep)
                        if dep_pos is None:
                            continue
                        dep_entities.update(
                            normalize_structure_text(entity)
                            for entity in doc_entities_for_pos(dep_pos)
                        )
                        dep_title = normalize_structure_text(pool_doc_titles[dep_pos])
                        if dep_title:
                            dep_entities.add(dep_title)
                    local_binding_candidates_by_req[req.unit_id] = collect_binding_candidates(
                        req,
                        dep_entities,
                        cover_positions=local_cover_position_by_req,
                    )
                    score, score_trace = score_requirement_for_position(
                        req,
                        pos,
                        dependency_entities=dep_entities,
                        binding_candidates_override=local_binding_candidates_by_req.get(req.unit_id, []),
                    )
                    previous_score = float(local_coverage_by_req.get(req.unit_id, 0.0))
                    previous_cover_position = local_cover_position_by_req.get(req.unit_id)
                    if score <= previous_score:
                        continue
                    local_coverage_by_req[req.unit_id] = score
                    should_set_cover = score >= float(match_threshold) and (
                        req.unit_id not in local_cover_position_by_req
                        or bool(enable_dependency_binding)
                    )
                    if should_set_cover:
                        local_cover_position_by_req[req.unit_id] = int(pos)
                        if previous_cover_position != int(pos):
                            made_progress = True
                        local_steps.append({
                            "unit_id": req.unit_id,
                            "pool_position": int(pos),
                            "title": pool_doc_titles[pos],
                            "coverage_score": round(float(score), 4),
                            "previous_coverage_score": round(float(previous_score), 4),
                            "cover_update": previous_cover_position is not None,
                            "binding_title": score_trace.get("binding_title", ""),
                        })
        covered_count = sum(
            1 for value in local_coverage_by_req.values()
            if float(value) >= float(match_threshold)
        )
        return {
            "positions": list(valid_positions),
            "covered_requirement_count": int(covered_count),
            "covered_requirement_rate": round(float(covered_count) / float(max(len(active_requirements), 1)), 4),
            "coverage_by_requirement": {
                req_id: round(float(score), 4)
                for req_id, score in local_coverage_by_req.items()
            },
            "cover_position_by_requirement": dict(local_cover_position_by_req),
            "coverage_steps": local_steps,
        }

    def repairable_requirement_status(req: EvidenceNeed) -> Dict[str, object]:
        has_explicit_anchor = bool(req.anchor_mentions)
        has_resolved_binding = bool(binding_candidates_by_req.get(req.unit_id))
        raw_inference_veto, inference_veto = _requirement_inference_veto(
            req,
            has_explicit_anchor=has_explicit_anchor,
            has_resolved_binding=has_resolved_binding,
            satisfiable_by_policy=normalized_satisfiable_by_policy,
        )
        satisfiable_by = _normalize_satisfiable_by(req.satisfiable_by)
        repairable = (has_explicit_anchor or has_resolved_binding) and not inference_veto
        if inference_veto:
            reason = "inference_only_veto"
        elif has_resolved_binding:
            reason = "resolved_dependency_binding"
        elif has_explicit_anchor:
            reason = "explicit_anchor"
        else:
            reason = "unresolved_dependency_or_unanchored"
        return {
            "repairable": bool(repairable),
            "reason": reason,
            "has_explicit_anchor": bool(has_explicit_anchor),
            "has_resolved_dependency_binding": bool(has_resolved_binding),
            "raw_inference_veto": bool(raw_inference_veto),
            "inference_veto": bool(inference_veto),
            "satisfiable_by": satisfiable_by,
            "satisfiable_by_policy": normalized_satisfiable_by_policy,
        }

    baseline_demand_assessment = compute_demand_assessment(range(target_k))
    demand_gate = {
        "enabled": bool(demand_gate_enabled),
        "alpha": round(float(demand_gate_alpha), 4),
        "baseline_covered_requirement_rate": float(
            baseline_demand_assessment.get("covered_requirement_rate", 0.0) or 0.0
        ),
        "keep_baseline": False,
        "reason": "disabled",
    }
    if bool(demand_gate_enabled):
        demand_gate["keep_baseline"] = (
            float(baseline_demand_assessment.get("covered_requirement_rate", 0.0) or 0.0)
            >= float(demand_gate_alpha)
        )
        demand_gate["reason"] = (
            "baseline_demand_satisfied"
            if bool(demand_gate["keep_baseline"])
            else "baseline_demand_incomplete"
        )
    if bool(demand_gate.get("keep_baseline", False)):
        baseline_positions = list(range(target_k))
        return baseline_positions, {
            "selection": "evidence_need_embed",
            "status": "demand_gate_keep",
            "query": str(query),
            "requirement_count": int(len(active_requirements)),
            "requirements": [req.to_trace() for req in active_requirements],
            "match_threshold": round(float(match_threshold), 4),
            "rank_weight": round(float(rank_weight), 4),
            "reserve_top_m": int(reserve_count),
            "dependency_binding_enabled": bool(enable_dependency_binding),
            "dependency_enforced": bool(enforce_dependencies),
            "binding_max_candidates": int(binding_max_candidates),
            "binding_entity_hit_required": bool(binding_entity_hit_required),
            "require_new_crossing": bool(require_new_crossing),
            "repairable_filter_enabled": bool(repairable_filter_enabled),
            "satisfiable_by_policy": normalized_satisfiable_by_policy,
            "demand_gate": demand_gate,
            "baseline_demand_assessment": baseline_demand_assessment,
            "binding_candidates_by_requirement": {},
            "covered_requirement_count": int(baseline_demand_assessment["covered_requirement_count"]),
            "covered_requirement_rate": float(baseline_demand_assessment["covered_requirement_rate"]),
            "coverage_by_requirement": dict(baseline_demand_assessment["coverage_by_requirement"]),
            "cover_position_by_requirement": dict(baseline_demand_assessment["cover_position_by_requirement"]),
            "selected_positions": baseline_positions,
            "selected_titles": [pool_doc_titles[pos] for pos in baseline_positions],
            "selection_steps": [],
            "embedding_available_requirement_count": int(sum(req_vectors.get(req.unit_id, np.array([])).size > 0 for req in active_requirements)),
            "embedding_available_doc_count": int(len(doc_vectors)),
        }

    ser_binding_candidates_by_req: Dict[str, List[Dict[str, object]]] = {}

    def ser_support_score(req: EvidenceNeed, pos: int) -> float:
        raw_score = float(raw_match_scores.get(req.unit_id, np.zeros(pool_limit))[pos])
        if not np.isfinite(raw_score):
            return 0.0
        threshold = float(match_threshold)
        if threshold >= 1.0:
            semantic_score = 1.0 if raw_score >= threshold else 0.0
        else:
            semantic_score = (raw_score - threshold) / max(1e-9, 1.0 - threshold)
        semantic_score = min(1.0, max(0.0, float(semantic_score)))
        binding_candidates = ser_binding_candidates_by_req.get(req.unit_id, [])
        if binding_candidates:
            best_binding_score = 0.0
            for candidate in binding_candidates:
                candidate_key = str(candidate.get("key", "") or "")
                hit_score = _binding_candidate_hit_score(
                    candidate_key,
                    doc_entities_for_pos(pos),
                    pool_doc_titles[pos],
                    pool_docs[pos],
                )
                bound_scores = ensure_bound_score(req, candidate)
                bound_score = float(bound_scores[pos]) if bound_scores is not None else 0.0
                if binding_entity_hit_required and hit_score <= 0.0:
                    effective = 0.0
                else:
                    effective = min(1.0, max(0.0, bound_score + float(dependency_bonus_weight) * hit_score))
                best_binding_score = max(best_binding_score, float(effective))
            semantic_score = best_binding_score
        if bool(ser_anchor_binding_enabled) and req.anchor_mentions:
            bind_score = _anchor_hit_score(
                req.anchor_mentions,
                pool_doc_titles[pos],
                pool_docs[pos],
                doc_entities_for_pos(pos),
            )
            semantic_score *= min(1.0, max(0.0, float(bind_score)))
        return min(1.0, max(0.0, semantic_score))

    def ser_coverage(positions: Sequence[int]) -> Dict[str, float]:
        valid_positions = [
            int(pos)
            for pos in positions
            if 0 <= int(pos) < pool_limit
        ]
        coverage: Dict[str, float] = {}
        for req in active_requirements:
            miss_prob = 1.0
            for pos in valid_positions:
                miss_prob *= 1.0 - ser_support_score(req, pos)
            coverage[req.unit_id] = min(1.0, max(0.0, 1.0 - miss_prob))
        return coverage

    def ser_objective(
        positions: Sequence[int],
        gap_by_req: Dict[str, float],
        lambda_q: float,
        baseline_position_set: set[int],
    ) -> Tuple[float, Dict[str, float], float, float]:
        coverage = ser_coverage(positions)
        repair_gain = sum(
            float(gap_by_req.get(req.unit_id, 0.0)) * float(coverage.get(req.unit_id, 0.0))
            for req in active_requirements
        )
        baseline_overlap = len({int(pos) for pos in positions} & baseline_position_set)
        preservation_gain = float(lambda_q) * float(baseline_overlap) / float(max(target_k, 1))
        return (
            float(repair_gain + preservation_gain),
            coverage,
            float(repair_gain),
            float(preservation_gain),
        )

    if bool(ser_enabled):
        baseline_positions = list(range(target_k))
        baseline_position_set = set(baseline_positions)
        baseline_cover_positions = {
            str(req_id): int(pos)
            for req_id, pos in dict(baseline_demand_assessment.get("cover_position_by_requirement", {})).items()
        }
        ser_binding_candidates_by_req.update({
            req.unit_id: collect_binding_candidates(req, set(), baseline_cover_positions)
            for req in active_requirements
        })
        repairable_by_req: Dict[str, Dict[str, object]] = {}
        for req in active_requirements:
            has_explicit_anchor = bool(req.anchor_mentions)
            has_resolved_binding = bool(ser_binding_candidates_by_req.get(req.unit_id))
            raw_inference_veto, inference_veto = _requirement_inference_veto(
                req,
                has_explicit_anchor=has_explicit_anchor,
                has_resolved_binding=has_resolved_binding,
                satisfiable_by_policy=normalized_satisfiable_by_policy,
            )
            satisfiable_by = _normalize_satisfiable_by(req.satisfiable_by)
            repairable = (has_explicit_anchor or has_resolved_binding) and not inference_veto
            if not bool(ser_repairable_gap_enabled):
                repairable = True
            if inference_veto:
                reason = "inference_only_veto"
            elif has_resolved_binding:
                reason = "resolved_dependency_binding"
            elif has_explicit_anchor:
                reason = "explicit_anchor"
            else:
                reason = "unresolved_dependency_or_unanchored"
            repairable_by_req[req.unit_id] = {
                "repairable": bool(repairable),
                "reason": reason,
                "has_explicit_anchor": bool(has_explicit_anchor),
                "has_resolved_dependency_binding": bool(has_resolved_binding),
                "raw_inference_veto": bool(raw_inference_veto),
                "inference_veto": bool(inference_veto),
                "satisfiable_by": satisfiable_by,
                "satisfiable_by_policy": normalized_satisfiable_by_policy,
                "binding_candidates": [
                    {
                        "title": str(row.get("title", "") or ""),
                        "dep": str(row.get("dep", "") or ""),
                        "title_pool_position": int(row.get("title_pool_position", -1)),
                    }
                    for row in ser_binding_candidates_by_req.get(req.unit_id, [])
                ],
            }
        baseline_ser_coverage = ser_coverage(baseline_positions)
        baseline_sufficiency = (
            sum(float(value) for value in baseline_ser_coverage.values())
            / float(max(len(active_requirements), 1))
        )
        raw_gap_by_req = {
            req.unit_id: max(0.0, 1.0 - float(baseline_ser_coverage.get(req.unit_id, 0.0)))
            for req in active_requirements
        }
        gap_by_req = {
            req.unit_id: (
                float(raw_gap_by_req.get(req.unit_id, 0.0))
                if bool(repairable_by_req.get(req.unit_id, {}).get("repairable", True))
                else 0.0
            )
            for req in active_requirements
        }
        lambda_q = max(0.0, float(ser_lambda0)) * float(baseline_sufficiency)
        current_positions = list(baseline_positions)
        current_objective, current_coverage, current_repair, current_preservation = ser_objective(
            current_positions,
            gap_by_req,
            lambda_q,
            baseline_position_set,
        )
        ser_steps: List[Dict[str, object]] = []
        max_swaps = max(0, min(target_k, pool_limit - target_k))
        for _ in range(max_swaps):
            selected_set = set(current_positions)
            best_swap: Dict[str, object] | None = None
            for candidate_pos in range(pool_limit):
                if candidate_pos in selected_set:
                    continue
                candidate_title_key = normalize_structure_text(pool_doc_titles[candidate_pos])
                for slot_idx, remove_pos in enumerate(list(current_positions)):
                    if bool(non_anchor_title_dedup) and candidate_title_key:
                        duplicate = any(
                            normalize_structure_text(pool_doc_titles[kept_pos]) == candidate_title_key
                            for kept_pos in current_positions
                            if int(kept_pos) != int(remove_pos)
                        )
                        if duplicate:
                            continue
                    proposed_positions = list(current_positions)
                    proposed_positions[slot_idx] = int(candidate_pos)
                    proposed_objective, proposed_coverage, proposed_repair, proposed_preservation = ser_objective(
                        proposed_positions,
                        gap_by_req,
                        lambda_q,
                        baseline_position_set,
                    )
                    delta = float(proposed_objective - current_objective)
                    if delta <= 1e-9:
                        continue
                    row = {
                        "slot": int(slot_idx),
                        "remove_position": int(remove_pos),
                        "remove_title": pool_doc_titles[remove_pos],
                        "add_position": int(candidate_pos),
                        "add_title": pool_doc_titles[candidate_pos],
                        "objective": float(proposed_objective),
                        "delta": float(delta),
                        "repair_gain": float(proposed_repair),
                        "preservation_gain": float(proposed_preservation),
                        "coverage": proposed_coverage,
                    }
                    if best_swap is None or (
                        float(row["delta"]),
                        float(row["objective"]),
                        int(remove_pos),
                        -int(candidate_pos),
                    ) > (
                        float(best_swap["delta"]),
                        float(best_swap["objective"]),
                        int(best_swap["remove_position"]),
                        -int(best_swap["add_position"]),
                    ):
                        best_swap = row
            if best_swap is None:
                break
            current_positions[int(best_swap["slot"])] = int(best_swap["add_position"])
            current_objective = float(best_swap["objective"])
            current_coverage = dict(best_swap["coverage"])
            current_repair = float(best_swap["repair_gain"])
            current_preservation = float(best_swap["preservation_gain"])
            ser_steps.append({
                "step": int(len(ser_steps) + 1),
                "mode": "ser_swap",
                "slot": int(best_swap["slot"]),
                "remove_position": int(best_swap["remove_position"]),
                "remove_title": str(best_swap["remove_title"]),
                "add_position": int(best_swap["add_position"]),
                "add_title": str(best_swap["add_title"]),
                "objective_delta": round(float(best_swap["delta"]), 6),
                "objective": round(float(current_objective), 6),
                "repair_gain": round(float(current_repair), 6),
                "preservation_gain": round(float(current_preservation), 6),
            })
        covered_count = sum(
            1 for value in current_coverage.values()
            if float(value) >= float(match_threshold)
        )
        trace = {
            "selection": "evidence_need_embed",
            "status": "ser_repair_applied" if ser_steps else "ser_repair_keep",
            "query": str(query),
            "requirement_count": int(len(active_requirements)),
            "requirements": [req.to_trace() for req in active_requirements],
            "match_threshold": round(float(match_threshold), 4),
            "rank_weight": round(float(rank_weight), 4),
            "reserve_top_m": int(reserve_count),
            "dependency_binding_enabled": bool(enable_dependency_binding),
            "dependency_enforced": bool(enforce_dependencies),
            "binding_max_candidates": int(binding_max_candidates),
            "binding_entity_hit_required": bool(binding_entity_hit_required),
            "require_new_crossing": bool(require_new_crossing),
            "repairable_filter_enabled": bool(repairable_filter_enabled),
            "satisfiable_by_policy": normalized_satisfiable_by_policy,
            "demand_gate": demand_gate,
            "baseline_demand_assessment": baseline_demand_assessment,
            "ser": {
                "enabled": True,
                "lambda0": round(float(ser_lambda0), 4),
                "lambda_q": round(float(lambda_q), 6),
                "anchor_binding_enabled": bool(ser_anchor_binding_enabled),
                "repairable_gap_enabled": bool(ser_repairable_gap_enabled),
                "baseline_sufficiency": round(float(baseline_sufficiency), 6),
                "baseline_coverage_by_requirement": {
                    req_id: round(float(score), 6)
                    for req_id, score in baseline_ser_coverage.items()
                },
                "raw_gap_by_requirement": {
                    req_id: round(float(score), 6)
                    for req_id, score in raw_gap_by_req.items()
                },
                "gap_by_requirement": {
                    req_id: round(float(score), 6)
                    for req_id, score in gap_by_req.items()
                },
                "repairable_by_requirement": repairable_by_req,
                "objective": round(float(current_objective), 6),
                "repair_gain": round(float(current_repair), 6),
                "preservation_gain": round(float(current_preservation), 6),
                "swap_count": int(len(ser_steps)),
            },
            "binding_candidates_by_requirement": {
                req_id: rows
                for req_id, rows in ser_binding_candidates_by_req.items()
                if rows
            },
            "covered_requirement_count": int(covered_count),
            "covered_requirement_rate": round(float(covered_count) / float(max(len(active_requirements), 1)), 4),
            "coverage_by_requirement": {
                req_id: round(float(score), 6)
                for req_id, score in current_coverage.items()
            },
            "cover_position_by_requirement": {
                req_id: int(max(
                    current_positions,
                    key=lambda pos, unit_id=req_id: ser_support_score(req_by_id[unit_id], int(pos)),
                ))
                for req_id in current_coverage
                if req_id in req_by_id and current_positions
            },
            "selected_positions": list(current_positions[:target_k]),
            "selected_titles": [pool_doc_titles[pos] for pos in current_positions[:target_k]],
            "selection_steps": ser_steps,
            "embedding_available_requirement_count": int(sum(req_vectors.get(req.unit_id, np.array([])).size > 0 for req in active_requirements)),
            "embedding_available_doc_count": int(len(doc_vectors)),
        }
        return current_positions[:target_k], trace

    for reserved_pos in list(selected_positions):
        mark_coverage_from_position(reserved_pos, mode="reserve")

    while len(selected_positions) < target_k:
        selected_set = set(selected_positions)
        best_row: Dict[str, object] | None = None
        for pos in range(pool_limit):
            if pos in selected_set:
                continue
            title_key = normalize_structure_text(pool_doc_titles[pos]) if pos < len(pool_doc_titles) else ""
            if non_anchor_title_dedup and title_key and title_key in seen_titles:
                continue
            doc_vec = doc_vectors.get(pos)
            redundancy = 0.0
            if doc_vec is not None and selected_positions:
                selected_sims = [
                    float(np.dot(doc_vec, doc_vectors[selected_pos]))
                    for selected_pos in selected_positions
                    if selected_pos in doc_vectors and doc_vectors[selected_pos].shape == doc_vec.shape
                ]
                redundancy = max(selected_sims) if selected_sims else 0.0
            coverage_gain = 0.0
            newly_crossed: List[str] = []
            req_rows: List[Dict[str, object]] = []
            for req in active_requirements:
                if any(dep not in cover_position_by_req for dep in req.depends_on):
                    continue
                repairable_status = repairable_requirement_status(req)
                if bool(repairable_filter_enabled) and not bool(repairable_status.get("repairable", False)):
                    continue
                req_score, score_trace = score_requirement_for_position(req, pos)
                previous = float(coverage_by_req.get(req.unit_id, 0.0))
                delta = max(0.0, req_score - previous)
                if req_score >= float(match_threshold) and previous < float(match_threshold):
                    newly_crossed.append(req.unit_id)
                coverage_gain += delta
                if delta > 0 or req_score >= float(match_threshold):
                    req_rows.append({
                        "unit_id": req.unit_id,
                        "score": round(float(req_score), 4),
                        "previous": round(float(previous), 4),
                        "delta": round(float(delta), 4),
                        "binding_title": score_trace.get("binding_title", ""),
                        "repairable": bool(repairable_status.get("repairable", False)),
                        "repairable_reason": str(repairable_status.get("reason", "")),
                    })
            total_gain = (
                float(coverage_gain)
                + 0.25 * float(len(newly_crossed))
                + float(base_weight) * float(normalized_base_scores[pos])
                - float(rank_weight) * float(normalized_ranks[pos])
                - float(redundancy_weight) * max(0.0, float(redundancy))
            )
            row = {
                "pool_position": int(pos),
                "title": pool_doc_titles[pos],
                "coverage_gain": float(coverage_gain),
                "new_requirement_count": int(len(newly_crossed)),
                "new_requirement_ids": list(newly_crossed),
                "base_score": float(normalized_base_scores[pos]),
                "normalized_rank": float(normalized_ranks[pos]),
                "rank_penalty": float(rank_weight) * float(normalized_ranks[pos]),
                "redundancy": float(redundancy),
                "total_gain": float(total_gain),
                "requirement_scores": req_rows,
            }
            if best_row is None or (
                float(row["total_gain"]),
                float(row["coverage_gain"]),
                int(row["new_requirement_count"]),
                -int(pos),
            ) > (
                float(best_row["total_gain"]),
                float(best_row["coverage_gain"]),
                int(best_row["new_requirement_count"]),
                -int(best_row["pool_position"]),
            ):
                best_row = row

        no_acceptable_candidate = best_row is None
        if best_row is not None:
            new_requirement_count = int(best_row["new_requirement_count"])
            if require_new_crossing:
                no_acceptable_candidate = new_requirement_count <= 0
            else:
                no_acceptable_candidate = (
                    float(best_row["coverage_gain"]) <= 1e-9
                    and new_requirement_count <= 0
                )

        if no_acceptable_candidate:
            fill_pos = next((pos for pos in range(pool_limit) if pos not in selected_set), None)
            if fill_pos is None:
                break
            selected_positions.append(int(fill_pos))
            title_key = normalize_structure_text(pool_doc_titles[fill_pos])
            if title_key:
                seen_titles.add(title_key)
            selection_steps.append({
                "step": int(len(selection_steps) + 1),
                "mode": "baseline_fill",
                "pool_position": int(fill_pos),
                "title": pool_doc_titles[fill_pos],
                "reason": (
                    "no_new_requirement_crossing"
                    if bool(require_new_crossing)
                    else "no_positive_requirement_gain"
                ),
            })
            continue

        chosen_pos = int(best_row["pool_position"])
        selected_positions.append(chosen_pos)
        title_key = normalize_structure_text(pool_doc_titles[chosen_pos])
        if title_key:
            seen_titles.add(title_key)
        selection_steps.append({
            "step": int(len(selection_steps) + 1),
            "mode": "evidence_need_cover",
            "pool_position": chosen_pos,
            "title": pool_doc_titles[chosen_pos],
            "coverage_gain": round(float(best_row["coverage_gain"]), 4),
            "new_requirement_count": int(best_row["new_requirement_count"]),
            "new_requirement_ids": list(best_row["new_requirement_ids"]),
            "base_score": round(float(best_row["base_score"]), 4),
            "normalized_rank": round(float(best_row["normalized_rank"]), 4),
            "rank_penalty": round(float(best_row["rank_penalty"]), 4),
            "redundancy": round(float(best_row["redundancy"]), 4),
            "total_gain": round(float(best_row["total_gain"]), 4),
            "requirement_scores": best_row["requirement_scores"],
        })
        mark_coverage_from_position(chosen_pos, mode="evidence_need_cover")

    if len(selected_positions) < target_k:
        selected_set = set(selected_positions)
        for pos in range(pool_limit):
            if pos in selected_set:
                continue
            selected_positions.append(pos)
            selected_set.add(pos)
            if len(selected_positions) >= target_k:
                break

    covered_count = sum(1 for value in coverage_by_req.values() if float(value) >= float(match_threshold))
    trace = {
        "selection": "evidence_need_embed",
        "status": "applied",
        "query": str(query),
        "requirement_count": int(len(active_requirements)),
        "requirements": [req.to_trace() for req in active_requirements],
        "match_threshold": round(float(match_threshold), 4),
        "rank_weight": round(float(rank_weight), 4),
        "reserve_top_m": int(reserve_count),
        "dependency_binding_enabled": bool(enable_dependency_binding),
        "dependency_enforced": bool(enforce_dependencies),
        "binding_max_candidates": int(binding_max_candidates),
        "binding_entity_hit_required": bool(binding_entity_hit_required),
        "require_new_crossing": bool(require_new_crossing),
        "repairable_filter_enabled": bool(repairable_filter_enabled),
        "satisfiable_by_policy": normalized_satisfiable_by_policy,
        "demand_gate": demand_gate,
        "baseline_demand_assessment": baseline_demand_assessment,
        "repairable_by_requirement": {
            req.unit_id: repairable_requirement_status(req)
            for req in active_requirements
        },
        "binding_candidates_by_requirement": {
            req_id: [
                {
                    "title": str(row.get("title", "") or ""),
                    "dep": str(row.get("dep", "") or ""),
                    "dep_position": int(row.get("dep_position", -1)),
                    "count": int(row.get("count", 0)),
                    "first_pos": int(row.get("first_pos", 10**9)),
                }
                for row in rows
            ]
            for req_id, rows in binding_candidates_by_req.items()
        },
        "covered_requirement_count": int(covered_count),
        "covered_requirement_rate": round(float(covered_count) / float(max(len(active_requirements), 1)), 4),
        "coverage_by_requirement": {
            req_id: round(float(score), 4)
            for req_id, score in coverage_by_req.items()
        },
        "cover_position_by_requirement": dict(cover_position_by_req),
        "selected_positions": list(selected_positions[:target_k]),
        "selected_titles": [pool_doc_titles[pos] for pos in selected_positions[:target_k]],
        "selection_steps": selection_steps,
        "embedding_available_requirement_count": int(sum(req_vectors.get(req.unit_id, np.array([])).size > 0 for req in active_requirements)),
        "embedding_available_doc_count": int(len(doc_vectors)),
    }
    return selected_positions[:target_k], trace
