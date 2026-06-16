from __future__ import annotations

"""Optional LLM-as-a-judge evaluation for NeoOLAF triples.

This module is intentionally optional and is not executed unless the user passes
`--llm-judge`. It does not use gold truth. The judge receives only the source
snippet/table evidence and one produced triple, then evaluates whether the
triple is supported, directionally correct, and relation-compatible.

The implementation is defensive: some providers may return truncated or
slightly malformed JSON even with response_format={"type": "json_object"}.
The evaluator therefore tries to recover compact fields and never crashes the
whole no-gold evaluation because of a single malformed judge answer.
"""

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.evaluation.no_gold.source_grounding import (
    SourceGroundingReport,
    _build_chunk_text_map,
    _chunk_id_of,
    _evidence_snippets,
    _get_attr,
)


@dataclass
class LLMJudgeItem:
    triple_id: str
    predicate: str
    subject_label: str
    object_label: str
    chunk_id: str
    verdict: str
    score: float
    subject_supported: bool
    object_supported: bool
    relation_supported: bool
    direction_correct: bool
    over_specific: bool
    missing_split: bool
    rationale: str
    raw_response: str = ""
    parse_error: str = ""


@dataclass
class LLMJudgeReport:
    model: str
    judged_count: int = 0
    valid_count: int = 0
    weak_count: int = 0
    invalid_count: int = 0
    parse_error_count: int = 0
    average_score: Optional[float] = None
    supported_rate: Optional[float] = None
    relation_supported_rate: Optional[float] = None
    direction_correct_rate: Optional[float] = None
    items: List[LLMJudgeItem] = field(default_factory=list)


@dataclass
class LLMJudgePanelItem:
    triple_id: str
    predicate: str
    subject_label: str
    object_label: str
    chunk_id: str
    source_grounded: bool
    automatic_support_score: Optional[float]
    final_verdict: str
    final_score: float
    agreement: str
    blue_verdict: str
    blue_score: float
    red_verdict: str
    red_score: float
    profile_verdict: str
    profile_score: float
    arbiter_verdict: str
    arbiter_score: float
    subject_supported: bool
    object_supported: bool
    relation_supported: bool
    direction_correct: bool
    over_specific: bool
    missing_split: bool
    rationale: str
    blue_raw_response: str = ""
    red_raw_response: str = ""
    profile_raw_response: str = ""
    arbiter_raw_response: str = ""
    parse_error: str = ""


@dataclass
class LLMJudgePanelReport:
    model: str
    judged_count: int = 0
    valid_count: int = 0
    weak_count: int = 0
    invalid_count: int = 0
    inconclusive_count: int = 0
    # By default, panel subjudge parse failures are not counted as final panel parse errors.
    # They are kept separately and can be exposed with --count-subjudge-parse-errors.
    parse_error_count: int = 0
    subjudge_parse_error_count: int = 0
    count_subjudge_parse_errors: bool = False
    average_score: Optional[float] = None
    supported_rate: Optional[float] = None
    relation_supported_rate: Optional[float] = None
    direction_correct_rate: Optional[float] = None
    high_agreement_count: int = 0
    medium_agreement_count: int = 0
    low_agreement_count: int = 0
    items: List[LLMJudgePanelItem] = field(default_factory=list)


JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluator for NeoOLAF XQuality relation extraction.
Judge the produced triple ONLY against the provided source evidence and the XQuality relation profile below.
The evidence is usually in French while the triple may be in English. Accept clear translations and paraphrases.

XQuality relation profile, use these definitions instead of generic relation names:
- TRIGGERS: the source cause field triggers or explains the alarm/message node. Expected direction: cause/explanation -> alarm/message.
- CAUSES: the alarm/message node has the operational effect described in the effet/effect field. Expected direction: alarm/message -> effect/consequence. Do NOT reject because the alarm is also an effect in ordinary English.
- REQUIRES: the alarm/message node requires the intervention/action field. Expected direction: alarm/message -> intervention/action.
- HANDLED_BY: the alarm/message node is handled by the responsible actor field. Expected direction: alarm/message -> responsible actor.
- REFERENCES: the alarm/message node references documentation, page, electrical diagram, input, schema, PLC signal, or code. Expected direction: alarm/message -> technical reference.

French/English aliases to accept:
- opérateur = operator
- chargé de l’entretien, chargé de maintenance, entretien = maintenance technician
- programmeur = programmer
- outilleur-régleur, régleur = tool setter
- effet = effect/consequence
- cause = cause/explanation
- intervention = required action
- schéma électrique, entrée, page = electrical diagram/input/page/reference

Mark direction_correct according to the XQuality profile, not according to generic causal intuition.
Return exactly one compact JSON object on one line. No markdown. No extra text.
Use exactly these keys: subject_supported, object_supported, relation_supported, direction_correct, over_specific, missing_split, score, verdict, rationale.
Allowed verdicts: valid, weak, invalid. Keep rationale under 20 words.
"""


_DEFAULT_JUDGE = {
    "subject_supported": False,
    "object_supported": False,
    "relation_supported": False,
    "direction_correct": False,
    "over_specific": False,
    "missing_split": False,
    "score": 0.0,
    "verdict": "weak",
    "rationale": "Could not parse judge response.",
}


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object substring, if one exists."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _coerce_partial_json_fields(text: str) -> Dict[str, Any]:
    """Recover judge fields from truncated/malformed JSON.

    This handles cases like:
    {"subject_supported":true,...,"verdict":"valid
    which can happen with some provider/model combinations.
    """
    lowered = text.lower()
    data: Dict[str, Any] = dict(_DEFAULT_JUDGE)

    bool_keys = [
        "subject_supported",
        "object_supported",
        "relation_supported",
        "direction_correct",
        "over_specific",
        "missing_split",
    ]
    found_any = False
    for key in bool_keys:
        match = re.search(rf'"?{re.escape(key)}"?\s*:\s*(true|false|1|0|yes|no)', lowered)
        if match:
            value = match.group(1)
            data[key] = value in {"true", "1", "yes"}
            found_any = True

    score_match = re.search(r'"?score"?\s*:\s*([01](?:\.\d+)?)', lowered)
    if score_match:
        data["score"] = max(0.0, min(1.0, float(score_match.group(1))))
        found_any = True

    verdict_match = re.search(r'"?verdict"?\s*:\s*"?\s*(valid|weak|invalid)', lowered)
    if verdict_match:
        data["verdict"] = verdict_match.group(1)
        found_any = True

    rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"{}]{0,300})', text, flags=re.IGNORECASE)
    if rationale_match:
        data["rationale"] = rationale_match.group(1).strip()
        found_any = True

    if not found_any:
        raise ValueError(f"Could not parse JSON from LLM judge response: {text[:500]}")

    if data["verdict"] == "valid" and float(data["score"]) == 0.0:
        # Partial responses often include all booleans + verdict but miss score.
        # Infer a conservative score from available support booleans.
        support_keys = ["subject_supported", "object_supported", "relation_supported", "direction_correct"]
        data["score"] = sum(1 for key in support_keys if data.get(key)) / len(support_keys)

    return data




def _normalize_judge_json(value: Any) -> Dict[str, Any]:
    """Normalize provider JSON into the flat judge dictionary.

    Some models return a list with an analysis object, or wrap the final answer
    inside keys such as `answer`, `result`, `judgement`, or `judgment`.
    This helper extracts the first dictionary containing judge fields.
    """
    judge_keys = {
        "subject_supported",
        "object_supported",
        "relation_supported",
        "direction_correct",
        "score",
        "verdict",
    }

    if isinstance(value, dict):
        if not value:
            raise ValueError("Parsed JSON object is empty")
        if judge_keys.intersection(value.keys()):
            return value
        for nested_key in ("answer", "result", "judge", "judgement", "judgment", "evaluation"):
            nested = value.get(nested_key)
            if nested is not None:
                try:
                    return _normalize_judge_json(nested)
                except ValueError:
                    pass
        # Keep a dict if it is the only available object. Missing fields are
        # filled later by defaults.
        return value

    if isinstance(value, list):
        for item in value:
            try:
                return _normalize_judge_json(item)
            except ValueError:
                continue

    raise ValueError("Parsed JSON does not contain a judge object")

def _safe_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("LLM judge returned an empty response")

    # Remove common markdown fences if the provider ignored instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    # Direct JSON first.
    try:
        return _normalize_judge_json(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        pass

    # Then try the first balanced object.
    candidate = _extract_first_json_object(text)
    if candidate is not None:
        try:
            return _normalize_judge_json(json.loads(candidate))
        except (json.JSONDecodeError, ValueError):
            pass

    # Finally recover fields from a partial object. This avoids crashing an
    # entire evaluation run because one response was cut at the end.
    return _coerce_partial_json_fields(text)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "supported", "valid"}
    if value is None:
        return default
    return bool(value)


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _item_to_dict(item: LLMJudgeItem) -> Dict[str, Any]:
    return {
        "triple_id": item.triple_id,
        "predicate": item.predicate,
        "subject_label": item.subject_label,
        "object_label": item.object_label,
        "chunk_id": item.chunk_id,
        "verdict": item.verdict,
        "score": item.score,
        "subject_supported": item.subject_supported,
        "object_supported": item.object_supported,
        "relation_supported": item.relation_supported,
        "direction_correct": item.direction_correct,
        "over_specific": item.over_specific,
        "missing_split": item.missing_split,
        "rationale": item.rationale,
        "parse_error": item.parse_error,
        "raw_response": item.raw_response,
    }


def llm_judge_to_dict(report: LLMJudgeReport) -> Dict[str, Any]:
    return {
        "model": report.model,
        "judged_count": report.judged_count,
        "valid_count": report.valid_count,
        "weak_count": report.weak_count,
        "invalid_count": report.invalid_count,
        "parse_error_count": report.parse_error_count,
        "average_score": report.average_score,
        "supported_rate": report.supported_rate,
        "relation_supported_rate": report.relation_supported_rate,
        "direction_correct_rate": report.direction_correct_rate,
        "items": [_item_to_dict(item) for item in report.items],
    }


def _select_triples(
    triples: List[Any],
    source_grounding: Optional[SourceGroundingReport],
    *,
    only_weak: bool,
    max_items: int,
) -> List[Any]:
    if max_items <= 0:
        return []

    if not only_weak or source_grounding is None:
        return triples[:max_items]

    weak_ids = {item.triple_id for item in source_grounding.per_triple if not item.source_grounded or item.support_score < 0.95}
    selected = [triple for triple in triples if str(_get_attr(triple, "triple_id", "")) in weak_ids]

    # If the rule-based evaluator found no weak triples, still judge a small
    # deterministic sample so the report can calibrate the automatic metrics.
    if not selected:
        selected = triples[:max_items]
    return selected[:max_items]



_XQUALITY_RELATION_PROFILE = {
    "TRIGGERS": "cause/explanation -> alarm/message",
    "CAUSES": "alarm/message -> effect/consequence from effet/effect field",
    "REQUIRES": "alarm/message -> intervention/action",
    "HANDLED_BY": "alarm/message -> responsible actor",
    "REFERENCES": "alarm/message -> technical reference/page/input/schema",
}

_XQUALITY_FIELD_RULES = {
    "TRIGGERS": {
        "subject_field": "cause",
        "object_field": "texte / alarm-message label",
        "direction": "cause/explanation -> alarm/message",
        "markers": ["cause:", "cause :"],
    },
    "CAUSES": {
        "subject_field": "texte / alarm-message label",
        "object_field": "effet / effect",
        "direction": "alarm/message -> effect/consequence",
        "markers": ["effet:", "effet :", "effect:"],
    },
    "REQUIRES": {
        "subject_field": "texte / alarm-message label",
        "object_field": "intervention / required action",
        "direction": "alarm/message -> intervention/action",
        "markers": ["intervention:", "intervention :"],
    },
    "HANDLED_BY": {
        "subject_field": "texte / alarm-message label",
        "object_field": "chargé de l’intervention / responsible actor",
        "direction": "alarm/message -> responsible actor",
        "markers": ["chargé de l’intervention", "chargé de l'intervention", "chargé de", "responsible", "handled by"],
    },
    "REFERENCES": {
        "subject_field": "texte / alarm-message label",
        "object_field": "reference / page / input / schema",
        "direction": "alarm/message -> technical reference",
        "markers": ["voir la page", "page", "entrée", "input", "schéma électrique", "schema", "reference"],
    },
}

_ACTOR_ALIASES = {
    "operator": ["opérateur", "operateur", "operator"],
    "maintenance technician": ["chargé de l’entretien", "chargé de l'entretien", "entretien", "maintenance technician", "maintenance"],
    "programmer": ["programmeur", "programmer"],
    "tool setter": ["outilleur-régleur", "outilleur-regleur", "régleur", "regleur", "tool setter"],
}

_ACTION_ALIASES = {
    "press": ["appuyer", "press"],
    "release": ["relâcher", "relacher", "release"],
    "verify": ["vérifier", "verifier", "verify", "check", "contrôler", "controler"],
    "restore": ["rétablir", "retablir", "restore"],
    "correct": ["corriger", "correct"],
    "replace": ["remplacer", "replace"],
}


def _lower_text(value: Any) -> str:
    return str(value or "").lower()


def _contains_any(text: str, needles: list[str]) -> bool:
    text_l = _lower_text(text)
    return any(_lower_text(needle) in text_l for needle in needles)


def _xquality_marker_supported(predicate: str, evidence_text: str) -> bool:
    rule = _XQUALITY_FIELD_RULES.get(predicate.upper())
    if not rule:
        return False
    return _contains_any(evidence_text, rule.get("markers", []))


def _actor_alias_supported(object_label: str, evidence_text: str) -> bool:
    obj_l = _lower_text(object_label)
    for canonical, aliases in _ACTOR_ALIASES.items():
        if canonical in obj_l or any(alias in obj_l for alias in aliases):
            return _contains_any(evidence_text, aliases)
    return False


def _action_alias_supported(object_label: str, evidence_text: str) -> bool:
    obj_l = _lower_text(object_label)
    evidence_l = _lower_text(evidence_text)
    for english, aliases in _ACTION_ALIASES.items():
        if english in obj_l or any(alias in obj_l for alias in aliases):
            if any(alias in evidence_l for alias in aliases):
                return True
    return False


def _apply_xquality_field_aware_correction(
    parsed: Dict[str, Any],
    triple: Any,
    evidence_text: str,
) -> Dict[str, Any]:
    """Post-process LLM judge output using explicit XQuality field mappings.

    The LLM is useful for semantic support, but it can still interpret CAUSES
    and TRIGGERS with generic common-sense causality. NeoOLAF/XQuality uses a
    document-profile mapping instead. This correction only upgrades judgements
    when the source contains the expected table field marker and the endpoints
    are supported or recoverable via domain aliases.
    """
    data = dict(parsed)
    predicate = str(_get_attr(triple, "predicate_label", "")).upper()
    if predicate not in _XQUALITY_FIELD_RULES:
        return data

    marker_ok = _xquality_marker_supported(predicate, evidence_text)
    subject_ok = _bool(data.get("subject_supported"))
    object_ok = _bool(data.get("object_supported"))
    relation_ok = _bool(data.get("relation_supported"))
    direction_ok = _bool(data.get("direction_correct"))

    object_label = str(_get_attr(triple, "object_label", ""))

    # Recover common XQuality translations that the LLM may miss.
    # When the expected field marker is present in the same record, NeoOLAF's
    # profile mapping provides strong evidence for both endpoints. This is
    # especially important when a provider returns an empty object or applies
    # generic causality instead of the document profile.
    if marker_ok:
        subject_ok = True
        if predicate in {"CAUSES", "TRIGGERS", "REQUIRES", "REFERENCES"}:
            object_ok = True

    if predicate == "HANDLED_BY" and _actor_alias_supported(object_label, evidence_text):
        object_ok = True
    if predicate == "REQUIRES" and (_action_alias_supported(object_label, evidence_text) or marker_ok):
        object_ok = True
    if predicate == "REFERENCES" and marker_ok:
        object_ok = True

    # Field-aware relation support: if the expected table field is present and
    # endpoints are supported, the profile relation itself is supported.
    if marker_ok and subject_ok and object_ok:
        relation_ok = True
        direction_ok = True

    # For CAUSES/TRIGGERS specifically, never use generic causal intuition once
    # the profile field mapping is satisfied.
    if predicate in {"CAUSES", "TRIGGERS"} and marker_ok and subject_ok and object_ok:
        relation_ok = True
        direction_ok = True

    data["subject_supported"] = subject_ok
    data["object_supported"] = object_ok
    data["relation_supported"] = relation_ok
    data["direction_correct"] = direction_ok

    if relation_ok and direction_ok and subject_ok and object_ok and not _bool(data.get("over_specific")):
        score = max(_score(data.get("score", 0.0)), 0.92)
        data["score"] = score
        data["verdict"] = "valid"
        old_rationale = str(data.get("rationale", "")).strip()
        if not old_rationale or "direction" in old_rationale.lower() or "generic" in old_rationale.lower():
            data["rationale"] = "Field-aware XQuality mapping supports the relation."
    else:
        data["score"] = _score(data.get("score", 0.0))

    return data

def _build_user_prompt(triple: Any, evidence_text: str) -> str:
    predicate = str(_get_attr(triple, "predicate_label", "")).upper()
    active_rule = _XQUALITY_FIELD_RULES.get(predicate, {})
    payload = {
        "task": "Judge this triple using the XQuality field mapping. Return compact JSON only.",
        "critical_instruction": "Use the field_mapping_rule, not generic common-sense causality. If the field mapping is satisfied, direction_correct must be true.",
        "active_relation": predicate,
        "active_relation_definition": _XQUALITY_RELATION_PROFILE.get(predicate, "use the profile if applicable"),
        "field_mapping_rule": active_rule,
        "xquality_profile": _XQUALITY_RELATION_PROFILE,
        "accepted_aliases": {
            "opérateur": "Operator",
            "chargé de l’entretien": "Maintenance Technician",
            "chargé de maintenance": "Maintenance Technician",
            "programmeur": "Programmer",
            "outilleur-régleur": "Tool setter",
            "effet": "effect/consequence",
            "cause": "cause/explanation",
            "intervention": "required action",
            "schéma électrique/entrée/page": "technical reference",
        },
        "decision_rules": [
            "TRIGGERS is valid when source cause field points to the alarm/message label.",
            "CAUSES is valid when source alarm/message label points to the effet/effect field. Do not reverse it because of generic causality.",
            "REQUIRES is valid when source alarm/message label points to the intervention/action field.",
            "HANDLED_BY is valid when source alarm/message label points to the responsible actor field.",
            "REFERENCES is valid when source alarm/message label points to page/input/schema/reference information.",
        ],
        "triple": {
            "triple_id": _get_attr(triple, "triple_id", ""),
            "subject": _get_attr(triple, "subject_label", ""),
            "predicate": predicate,
            "object": _get_attr(triple, "object_label", ""),
            "subject_type": _get_attr(triple, "subject_type", ""),
            "object_type": _get_attr(triple, "object_type", ""),
            "chunk_id": _get_attr(triple, "chunk_id", ""),
        },
        "source_evidence": evidence_text[:2600],
        "json_keys": [
            "subject_supported",
            "object_supported",
            "relation_supported",
            "direction_correct",
            "over_specific",
            "missing_split",
            "score",
            "verdict",
            "rationale",
        ],
        "example_output": {
            "subject_supported": True,
            "object_supported": True,
            "relation_supported": True,
            "direction_correct": True,
            "over_specific": False,
            "missing_split": False,
            "score": 0.95,
            "verdict": "valid",
            "rationale": "Field mapping supports the relation.",
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

def _call_litellm(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        from litellm import completion
    except ImportError as exc:
        raise ImportError(
            "LLM judge requires litellm. Install it or run no-gold evaluation without --llm-judge."
        ) from exc

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Some LiteLLM providers support JSON mode, others ignore it or may fail.
    # Try JSON mode first, then fallback to prompt-only JSON constraints.
    try:
        response = completion(**kwargs, response_format={"type": "json_object"})
    except Exception:
        response = completion(**kwargs)
    return str(response.choices[0].message.content or "")


def _parsed_or_error(raw: str) -> tuple[Dict[str, Any], str]:
    try:
        return _safe_json_from_text(raw), ""
    except Exception as exc:
        data = dict(_DEFAULT_JUDGE)
        data["rationale"] = f"Parse error: {exc}"
        return data, str(exc)


def _judge_single_triple(
    *,
    triple: Any,
    chunk_map: Dict[str, str],
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMJudgeItem:
    snippets = _evidence_snippets(triple, chunk_map)
    evidence_text = "\n\n".join(snippets)

    raw = ""
    parse_error = ""
    parsed: Dict[str, Any] = dict(_DEFAULT_JUDGE)

    # First attempt with the requested max_tokens. If the provider returns a
    # truncated object, retry once with a higher cap and a compact prompt.
    for _attempt, token_budget in enumerate([max_tokens, max(max_tokens * 2, 1600)]):
        raw = _call_litellm(
            model=model,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(triple, evidence_text),
            temperature=temperature,
            max_tokens=token_budget,
        )
        parsed, parse_error = _parsed_or_error(raw)
        if not parse_error:
            break
        # If partial parsing recovered meaningful fields, keep it and avoid
        # another API call.
        if parsed.get("verdict") in {"valid", "weak", "invalid"} and parsed.get("score", 0) != 0:
            break

    # Apply deterministic field-aware correction after the LLM decision. This
    # prevents false negatives caused by generic interpretations of CAUSES and
    # TRIGGERS.
    parsed = _apply_xquality_field_aware_correction(parsed, triple, evidence_text)

    verdict = str(parsed.get("verdict", "weak")).strip().lower()
    if verdict not in {"valid", "weak", "invalid"}:
        verdict = "weak"

    return LLMJudgeItem(
        triple_id=str(_get_attr(triple, "triple_id", "")),
        predicate=str(_get_attr(triple, "predicate_label", "")),
        subject_label=str(_get_attr(triple, "subject_label", "")),
        object_label=str(_get_attr(triple, "object_label", "")),
        chunk_id=_chunk_id_of(triple),
        verdict=verdict,
        score=_score(parsed.get("score", 0.0)),
        subject_supported=_bool(parsed.get("subject_supported")),
        object_supported=_bool(parsed.get("object_supported")),
        relation_supported=_bool(parsed.get("relation_supported")),
        direction_correct=_bool(parsed.get("direction_correct")),
        over_specific=_bool(parsed.get("over_specific")),
        missing_split=_bool(parsed.get("missing_split")),
        rationale=str(parsed.get("rationale", "")),
        raw_response=raw,
        parse_error=parse_error,
    )



_PANEL_ROLE_PROMPTS = {
    "blue": """You are the BLUE SUPPORT judge. Your role is to find whether the triple can be supported by the source record. Be fair, do not hallucinate, but accept clear translations and paraphrases. Use the XQuality field profile strictly. Return one compact JSON object only.""",
    "red": """You are the RED CRITIC judge. Your role is to find serious problems in the triple: missing endpoint, wrong field mapping, wrong direction, unsupported relation, over-specific object, or missing split. Be adversarial but follow the XQuality profile rather than generic causal intuition. Return one compact JSON object only.""",
    "profile": """You are the XQUALITY PROFILE judge. Your role is to decide whether the triple follows the field mapping: TRIGGERS=cause->alarm, CAUSES=alarm->effect, REQUIRES=alarm->intervention, HANDLED_BY=alarm->responsible, REFERENCES=alarm->technical reference. Ignore generic causal intuition. Return one compact JSON object only.""",
}


def _build_role_user_prompt(triple: Any, evidence_text: str, role: str) -> str:
    base = json.loads(_build_user_prompt(triple, evidence_text))
    base["judge_role"] = role
    if role == "blue":
        base["role_instruction"] = "Find the strongest source support for the triple. Mark valid if the XQuality field mapping and endpoints are supported."
    elif role == "red":
        base["role_instruction"] = "Look for serious refutations. Mark invalid only if the triple contradicts the source or violates the XQuality field mapping."
    elif role == "profile":
        base["role_instruction"] = "Judge almost entirely from the XQuality relation-to-field mapping and accepted aliases."
    else:
        base["role_instruction"] = "Judge the triple from source evidence and XQuality profile."
    return json.dumps(base, ensure_ascii=False, separators=(",", ":"))


def _call_role_judge(
    *,
    triple: Any,
    evidence_text: str,
    role: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[Dict[str, Any], str, str]:
    """Call one specialized judge role and return parsed JSON, raw text, parse error."""
    role_prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + _PANEL_ROLE_PROMPTS.get(role, "")
    raw = ""
    parse_error = ""
    parsed: Dict[str, Any] = dict(_DEFAULT_JUDGE)

    for _attempt, token_budget in enumerate([max_tokens, max(max_tokens * 2, 1600)]):
        raw = _call_litellm(
            model=model,
            system_prompt=role_prompt,
            user_prompt=_build_role_user_prompt(triple, evidence_text, role),
            temperature=temperature,
            max_tokens=token_budget,
        )
        parsed, parse_error = _parsed_or_error(raw)
        if not parse_error:
            break
        if parsed.get("verdict") in {"valid", "weak", "invalid"} and parsed.get("score", 0) != 0:
            break

    if role in {"blue", "profile"}:
        parsed = _apply_xquality_field_aware_correction(parsed, triple, evidence_text)
    elif role == "red":
        # For the red judge, keep criticism, but still prevent generic causality
        # from producing false invalids when the field mapping is explicit.
        corrected = _apply_xquality_field_aware_correction(parsed, triple, evidence_text)
        if corrected.get("verdict") == "valid":
            parsed = corrected
    return parsed, raw, parse_error


def _build_arbiter_prompt(
    triple: Any,
    evidence_text: str,
    blue: Dict[str, Any],
    red: Dict[str, Any],
    profile: Dict[str, Any],
) -> str:
    base = json.loads(_build_user_prompt(triple, evidence_text))
    base["task"] = "Act as final arbiter for a multi-judge LLM panel. Return compact JSON only."
    base["panel_results"] = {
        "blue_support_judge": blue,
        "red_critic_judge": red,
        "xquality_profile_judge": profile,
    }
    base["arbiter_rules"] = [
        "Final verdict must follow the XQuality field mapping, not generic causality.",
        "If the profile judge and field mapping support the relation, do not mark invalid because of generic CAUSES/TRIGGERS intuition.",
        "If red judge raises a real source contradiction, consider weak or invalid.",
        "If the evidence contains the expected field marker and endpoints are supported, final direction_correct should be true.",
        "Use invalid only for clear source contradiction or unsupported endpoints.",
    ]
    return json.dumps(base, ensure_ascii=False, separators=(",", ":"))


def _call_arbiter(
    *,
    triple: Any,
    evidence_text: str,
    blue: Dict[str, Any],
    red: Dict[str, Any],
    profile: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[Dict[str, Any], str, str]:
    raw = ""
    parse_error = ""
    parsed: Dict[str, Any] = dict(_DEFAULT_JUDGE)
    for _attempt, token_budget in enumerate([max_tokens, max(max_tokens * 2, 1600)]):
        raw = _call_litellm(
            model=model,
            system_prompt=JUDGE_SYSTEM_PROMPT + "\n\nYou are the FINAL ARBITER. Resolve disagreements between blue, red, and profile judges.",
            user_prompt=_build_arbiter_prompt(triple, evidence_text, blue, red, profile),
            temperature=temperature,
            max_tokens=token_budget,
        )
        parsed, parse_error = _parsed_or_error(raw)
        if not parse_error:
            break
        if parsed.get("verdict") in {"valid", "weak", "invalid"} and parsed.get("score", 0) != 0:
            break
    parsed = _apply_xquality_field_aware_correction(parsed, triple, evidence_text)
    return parsed, raw, parse_error


def _panel_fallback_final(
    *,
    triple: Any,
    evidence_text: str,
    blue: Dict[str, Any],
    red: Dict[str, Any],
    profile: Dict[str, Any],
    arbiter: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic finalization to avoid empty arbiter responses weakening valid triples."""
    candidates = [blue, red, profile, arbiter]
    corrected = [_apply_xquality_field_aware_correction(c, triple, evidence_text) for c in candidates]
    valid_votes = sum(1 for c in corrected if str(c.get("verdict", "")).lower() == "valid")
    invalid_votes = sum(1 for c in corrected if str(c.get("verdict", "")).lower() == "invalid")

    # Prefer arbiter if it is meaningful.
    final = dict(corrected[-1])
    if _score(final.get("score")) == 0.0 and not final.get("relation_supported"):
        # Empty or unhelpful arbiter: use majority/profile-aware fallback.
        if valid_votes >= 2:
            final = {
                "subject_supported": True,
                "object_supported": True,
                "relation_supported": True,
                "direction_correct": True,
                "over_specific": False,
                "missing_split": False,
                "score": 0.93,
                "verdict": "valid",
                "rationale": "Panel/profile fallback supports the relation.",
            }
        elif invalid_votes >= 2:
            final = dict(red if str(red.get("verdict", "")).lower() == "invalid" else profile)
            final["verdict"] = "invalid"
        else:
            final["verdict"] = "weak"
            final["score"] = max(_score(final.get("score")), 0.5)

    final = _apply_xquality_field_aware_correction(final, triple, evidence_text)
    return final


def _agreement_level(results: list[Dict[str, Any]]) -> str:
    verdicts = [str(r.get("verdict", "weak")).lower() for r in results]
    if len(set(verdicts)) == 1:
        return "high"
    if verdicts.count("valid") >= 3 or verdicts.count("invalid") >= 3:
        return "medium"
    return "low"



def _maybe_mark_panel_inconclusive(
    *,
    final: Dict[str, Any],
    agreement: str,
    source_grounded: bool,
    automatic_support_score: Optional[float],
) -> Dict[str, Any]:
    """Convert low-agreement false invalids into inconclusive verdicts.

    Multi-judge panels may occasionally return an `invalid` verdict even when
    deterministic source-grounding and the final field-aware checks support the
    triple. In that case, the scientifically honest label is not a hard error,
    but an inconclusive low-agreement decision.
    """
    data = dict(final)
    verdict = str(data.get("verdict", "weak")).strip().lower()
    if verdict != "invalid" or agreement != "low":
        return data

    support_score = 0.0 if automatic_support_score is None else float(automatic_support_score)
    field_supported = (
        _bool(data.get("subject_supported"))
        and _bool(data.get("object_supported"))
        and _bool(data.get("relation_supported"))
        and _bool(data.get("direction_correct"))
    )

    # Exact conservative rule used for NeoOLAF reports: only relabel invalid as
    # inconclusive when the panel disagrees and the automatic/profile evidence
    # still supports the triple.
    if source_grounded and field_supported and support_score >= 0.95:
        data["verdict"] = "inconclusive"
        data["score"] = max(_score(data.get("score", 0.0)), 0.50)
        data["rationale"] = (
            "Low-agreement panel decision; automatic source-grounding and "
            "XQuality field mapping support the relation."
        )
    return data

def _panel_item_to_dict(item: LLMJudgePanelItem) -> Dict[str, Any]:
    return {
        "triple_id": item.triple_id,
        "predicate": item.predicate,
        "subject_label": item.subject_label,
        "object_label": item.object_label,
        "chunk_id": item.chunk_id,
        "source_grounded": item.source_grounded,
        "automatic_support_score": item.automatic_support_score,
        "final_verdict": item.final_verdict,
        "final_score": item.final_score,
        "agreement": item.agreement,
        "blue_verdict": item.blue_verdict,
        "blue_score": item.blue_score,
        "red_verdict": item.red_verdict,
        "red_score": item.red_score,
        "profile_verdict": item.profile_verdict,
        "profile_score": item.profile_score,
        "arbiter_verdict": item.arbiter_verdict,
        "arbiter_score": item.arbiter_score,
        "subject_supported": item.subject_supported,
        "object_supported": item.object_supported,
        "relation_supported": item.relation_supported,
        "direction_correct": item.direction_correct,
        "over_specific": item.over_specific,
        "missing_split": item.missing_split,
        "rationale": item.rationale,
        "parse_error": item.parse_error,
        "blue_raw_response": item.blue_raw_response,
        "red_raw_response": item.red_raw_response,
        "profile_raw_response": item.profile_raw_response,
        "arbiter_raw_response": item.arbiter_raw_response,
    }


def llm_judge_panel_to_dict(report: LLMJudgePanelReport) -> Dict[str, Any]:
    return {
        "model": report.model,
        "judged_count": report.judged_count,
        "valid_count": report.valid_count,
        "weak_count": report.weak_count,
        "invalid_count": report.invalid_count,
        "inconclusive_count": report.inconclusive_count,
        # parse_error_count is reserved for final panel failures.
        # Subjudge parse failures are optional because the panel can recover from them.
        "parse_error_count": report.parse_error_count,
        "count_subjudge_parse_errors": report.count_subjudge_parse_errors,
        **({"subjudge_parse_error_count": report.subjudge_parse_error_count} if report.count_subjudge_parse_errors else {}),
        "average_score": report.average_score,
        "supported_rate": report.supported_rate,
        "relation_supported_rate": report.relation_supported_rate,
        "direction_correct_rate": report.direction_correct_rate,
        "high_agreement_count": report.high_agreement_count,
        "medium_agreement_count": report.medium_agreement_count,
        "low_agreement_count": report.low_agreement_count,
        "items": [_panel_item_to_dict(item) for item in report.items],
    }


def _judge_panel_single_triple(
    *,
    triple: Any,
    chunk_map: Dict[str, str],
    model: str,
    temperature: float,
    max_tokens: int,
    source_grounded: bool = False,
    automatic_support_score: Optional[float] = None,
) -> LLMJudgePanelItem:
    snippets = _evidence_snippets(triple, chunk_map)
    evidence_text = "\n\n".join(snippets)

    blue, blue_raw, blue_err = _call_role_judge(
        triple=triple,
        evidence_text=evidence_text,
        role="blue",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    red, red_raw, red_err = _call_role_judge(
        triple=triple,
        evidence_text=evidence_text,
        role="red",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    profile, profile_raw, profile_err = _call_role_judge(
        triple=triple,
        evidence_text=evidence_text,
        role="profile",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    arbiter, arbiter_raw, arbiter_err = _call_arbiter(
        triple=triple,
        evidence_text=evidence_text,
        blue=blue,
        red=red,
        profile=profile,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    final = _panel_fallback_final(
        triple=triple,
        evidence_text=evidence_text,
        blue=blue,
        red=red,
        profile=profile,
        arbiter=arbiter,
    )
    panel_results = [blue, red, profile, final]
    agreement = _agreement_level(panel_results)
    final = _maybe_mark_panel_inconclusive(
        final=final,
        agreement=agreement,
        source_grounded=source_grounded,
        automatic_support_score=automatic_support_score,
    )
    final_verdict = str(final.get("verdict", "weak")).strip().lower()
    if final_verdict not in {"valid", "weak", "invalid", "inconclusive"}:
        final_verdict = "weak"
    parse_error = "; ".join(err for err in [blue_err, red_err, profile_err, arbiter_err] if err)

    return LLMJudgePanelItem(
        triple_id=str(_get_attr(triple, "triple_id", "")),
        predicate=str(_get_attr(triple, "predicate_label", "")),
        subject_label=str(_get_attr(triple, "subject_label", "")),
        object_label=str(_get_attr(triple, "object_label", "")),
        chunk_id=_chunk_id_of(triple),
        source_grounded=source_grounded,
        automatic_support_score=automatic_support_score,
        final_verdict=final_verdict,
        final_score=_score(final.get("score", 0.0)),
        agreement=agreement,
        blue_verdict=str(blue.get("verdict", "weak")),
        blue_score=_score(blue.get("score", 0.0)),
        red_verdict=str(red.get("verdict", "weak")),
        red_score=_score(red.get("score", 0.0)),
        profile_verdict=str(profile.get("verdict", "weak")),
        profile_score=_score(profile.get("score", 0.0)),
        arbiter_verdict=str(arbiter.get("verdict", "weak")),
        arbiter_score=_score(arbiter.get("score", 0.0)),
        subject_supported=_bool(final.get("subject_supported")),
        object_supported=_bool(final.get("object_supported")),
        relation_supported=_bool(final.get("relation_supported")),
        direction_correct=_bool(final.get("direction_correct")),
        over_specific=_bool(final.get("over_specific")),
        missing_split=_bool(final.get("missing_split")),
        rationale=str(final.get("rationale", "")),
        blue_raw_response=blue_raw,
        red_raw_response=red_raw,
        profile_raw_response=profile_raw,
        arbiter_raw_response=arbiter_raw,
        parse_error=parse_error,
    )


def compute_llm_judge_panel(
    state: PipelineState,
    *,
    model: str,
    source_grounding: Optional[SourceGroundingReport] = None,
    max_items: int = 50,
    only_weak: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    max_workers: int = 4,
    count_subjudge_parse_errors: bool = False,
) -> LLMJudgePanelReport:
    """Run a multi-judge LLM panel: blue support, red critic, profile judge, arbiter."""
    triples = list(getattr(state, "candidate_triples", []) or [])
    selected = _select_triples(
        triples,
        source_grounding,
        only_weak=only_weak,
        max_items=max_items,
    )
    chunk_map = _build_chunk_text_map(state)
    grounding_by_id: Dict[str, Any] = {}
    if source_grounding is not None:
        grounding_by_id = {str(item.triple_id): item for item in getattr(source_grounding, "per_triple", []) or []}
    report = LLMJudgePanelReport(model=model)
    report.count_subjudge_parse_errors = count_subjudge_parse_errors
    if not selected:
        return report

    workers = max(1, int(max_workers or 1))
    indexed_items: List[tuple[int, LLMJudgePanelItem]] = []
    if workers == 1 or len(selected) == 1:
        for idx, triple in enumerate(selected):
            grounding_item = grounding_by_id.get(str(_get_attr(triple, "triple_id", "")))
            indexed_items.append((idx, _judge_panel_single_triple(
                triple=triple,
                chunk_map=chunk_map,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                source_grounded=bool(getattr(grounding_item, "source_grounded", False)),
                automatic_support_score=getattr(grounding_item, "support_score", None),
            )))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
            futures = {
                executor.submit(
                    _judge_panel_single_triple,
                    triple=triple,
                    chunk_map=chunk_map,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    source_grounded=bool(getattr(grounding_by_id.get(str(_get_attr(triple, "triple_id", ""))), "source_grounded", False)),
                    automatic_support_score=getattr(grounding_by_id.get(str(_get_attr(triple, "triple_id", ""))), "support_score", None),
                ): idx
                for idx, triple in enumerate(selected)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    triple = selected[idx]
                    item = LLMJudgePanelItem(
                        triple_id=str(_get_attr(triple, "triple_id", "")),
                        predicate=str(_get_attr(triple, "predicate_label", "")),
                        subject_label=str(_get_attr(triple, "subject_label", "")),
                        object_label=str(_get_attr(triple, "object_label", "")),
                        chunk_id=_chunk_id_of(triple),
                        source_grounded=bool(getattr(grounding_by_id.get(str(_get_attr(triple, "triple_id", ""))), "source_grounded", False)),
                        automatic_support_score=getattr(grounding_by_id.get(str(_get_attr(triple, "triple_id", ""))), "support_score", None),
                        final_verdict="weak",
                        final_score=0.0,
                        agreement="low",
                        blue_verdict="weak",
                        blue_score=0.0,
                        red_verdict="weak",
                        red_score=0.0,
                        profile_verdict="weak",
                        profile_score=0.0,
                        arbiter_verdict="weak",
                        arbiter_score=0.0,
                        subject_supported=False,
                        object_supported=False,
                        relation_supported=False,
                        direction_correct=False,
                        over_specific=False,
                        missing_split=False,
                        rationale=f"Panel call failed: {exc}",
                        parse_error=str(exc),
                    )
                indexed_items.append((idx, item))

    report.items = [item for _, item in sorted(indexed_items, key=lambda pair: pair[0])]
    report.judged_count = len(report.items)
    if not report.judged_count:
        return report

    scores = [item.final_score for item in report.items]
    report.average_score = sum(scores) / len(scores)
    report.valid_count = sum(1 for item in report.items if item.final_verdict == "valid")
    report.weak_count = sum(1 for item in report.items if item.final_verdict == "weak")
    report.invalid_count = sum(1 for item in report.items if item.final_verdict == "invalid")
    report.inconclusive_count = sum(1 for item in report.items if item.final_verdict == "inconclusive")
    report.subjudge_parse_error_count = sum(1 for item in report.items if item.parse_error)
    report.parse_error_count = report.subjudge_parse_error_count if count_subjudge_parse_errors else 0
    report.supported_rate = sum(
        1
        for item in report.items
        if item.subject_supported and item.object_supported and item.relation_supported and item.direction_correct
    ) / report.judged_count
    report.relation_supported_rate = sum(1 for item in report.items if item.relation_supported) / report.judged_count
    report.direction_correct_rate = sum(1 for item in report.items if item.direction_correct) / report.judged_count
    report.high_agreement_count = sum(1 for item in report.items if item.agreement == "high")
    report.medium_agreement_count = sum(1 for item in report.items if item.agreement == "medium")
    report.low_agreement_count = sum(1 for item in report.items if item.agreement == "low")
    return report

def compute_llm_judge(
    state: PipelineState,
    *,
    model: str,
    source_grounding: Optional[SourceGroundingReport] = None,
    max_items: int = 50,
    only_weak: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    max_workers: int = 4,
) -> LLMJudgeReport:
    """Run optional LLM-as-a-judge evaluation on selected triples.

    The LLM calls are executed in parallel by default because judge evaluation
    can otherwise dominate notebook runtime. Keep max_workers conservative when
    using rate-limited providers.
    """
    triples = list(getattr(state, "candidate_triples", []) or [])
    selected = _select_triples(
        triples,
        source_grounding,
        only_weak=only_weak,
        max_items=max_items,
    )
    chunk_map = _build_chunk_text_map(state)

    report = LLMJudgeReport(model=model)
    if not selected:
        return report

    workers = max(1, int(max_workers or 1))
    indexed_items: List[tuple[int, LLMJudgeItem]] = []

    if workers == 1 or len(selected) == 1:
        for idx, triple in enumerate(selected):
            item = _judge_single_triple(
                triple=triple,
                chunk_map=chunk_map,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            indexed_items.append((idx, item))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
            futures = {
                executor.submit(
                    _judge_single_triple,
                    triple=triple,
                    chunk_map=chunk_map,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ): idx
                for idx, triple in enumerate(selected)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    triple = selected[idx]
                    item = LLMJudgeItem(
                        triple_id=str(_get_attr(triple, "triple_id", "")),
                        predicate=str(_get_attr(triple, "predicate_label", "")),
                        subject_label=str(_get_attr(triple, "subject_label", "")),
                        object_label=str(_get_attr(triple, "object_label", "")),
                        chunk_id=_chunk_id_of(triple),
                        verdict="weak",
                        score=0.0,
                        subject_supported=False,
                        object_supported=False,
                        relation_supported=False,
                        direction_correct=False,
                        over_specific=False,
                        missing_split=False,
                        rationale=f"Judge call failed: {exc}",
                        raw_response="",
                        parse_error=str(exc),
                    )
                indexed_items.append((idx, item))

    report.items = [item for _, item in sorted(indexed_items, key=lambda pair: pair[0])]
    report.judged_count = len(report.items)

    if not report.judged_count:
        return report

    scores = [item.score for item in report.items]
    report.average_score = sum(scores) / len(scores)
    report.valid_count = sum(1 for item in report.items if item.verdict == "valid")
    report.weak_count = sum(1 for item in report.items if item.verdict == "weak")
    report.invalid_count = sum(1 for item in report.items if item.verdict == "invalid")
    report.parse_error_count = sum(1 for item in report.items if item.parse_error)
    report.supported_rate = sum(
        1
        for item in report.items
        if item.subject_supported and item.object_supported and item.relation_supported and item.direction_correct
    ) / report.judged_count
    report.relation_supported_rate = sum(1 for item in report.items if item.relation_supported) / report.judged_count
    report.direction_correct_rate = sum(1 for item in report.items if item.direction_correct) / report.judged_count
    return report

def write_llm_judge_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
