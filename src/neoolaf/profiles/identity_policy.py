from __future__ import annotations

"""Generic, profile-driven record identity policy.

This module prevents dataset-specific rules from being hard-coded in Layer 1.
Profiles define what record types exist, how identifiers are hinted, and whether
LLM identifiers should be overridden by the current structural unit.
"""

import re
from typing import Any


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_number(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\d{2,8}", str(text))
    return match.group(0) if match else None


def _normalize_regex_pattern(pattern: Any) -> str:
    """Return a regex pattern tolerant to over-escaped JSON profiles.

    Some profiles may accidentally contain doubled regex escapes such as
    ``\\b`` instead of ``\b`` after JSON loading. Normalizing here keeps
    the identity policy robust and avoids dataset-specific fixes in Layer 1.
    """
    text = str(pattern or "")
    # JSON profiles should load regexes as "\b", but an older profile version
    # produced "\\b". Collapse doubled backslashes only when they are present.
    if "\\\\" in text:
        text = text.replace("\\\\", "\\")
    return text


def _policy(profile_config: dict[str, Any] | None) -> dict[str, Any]:
    profile_config = profile_config or {}
    policy = profile_config.get("record_identity_policy") or {}
    return policy if isinstance(policy, dict) else {}


def infer_record_identity_from_unit(
    unit: dict[str, Any] | None,
    profile_config: dict[str, Any] | None,
) -> dict[str, str | None]:
    """Infer record identity from a compact structural unit and its profile.

    This is generic: the profile supplies record types, hint fields, id fields,
    and header patterns. XQuality may define alarm/message, while another
    dataset can define event/incident/action.
    """
    unit = unit or {}
    policy = _policy(profile_config)

    # Direct hint fields are the most reliable source of current identity.
    record_type_hint = _first_non_empty(unit.get("record_type_hint"))
    record_id_hint = _first_non_empty(unit.get("record_id_hint"))

    if record_type_hint or record_id_hint:
        result = {
            "record_type": record_type_hint,
            "record_id": record_id_hint,
        }
        for record_type, cfg in (policy.get("record_types") or {}).items():
            hint_field = cfg.get("hint_field")
            id_field = cfg.get("id_field")
            value = _first_non_empty(unit.get(hint_field), unit.get(id_field)) if hint_field or id_field else None
            if value:
                result["record_type"] = record_type_hint or record_type
                result["record_id"] = record_id_hint or value
                result[id_field] = value
        return result

    # Pattern-based fallback over current header/title fields and row headers.
    haystacks: list[str] = []
    for key in ["current_header_text", "title", "section_key", "subsection_key", "unit_id"]:
        value = unit.get(key)
        if value:
            haystacks.append(str(value))
    record_identity_source = unit.get("record_identity_source") or {}
    if isinstance(record_identity_source, dict):
        for value in record_identity_source.values():
            if value:
                haystacks.append(str(value))
    for row in unit.get("field_value_rows") or []:
        if isinstance(row, dict):
            haystacks.append(str(row.get("field") or ""))
            haystacks.extend(str(v) for v in row.get("values") or [])
    haystack = "\n".join(haystacks)

    for record_type, cfg in (policy.get("record_types") or {}).items():
        id_field = cfg.get("id_field")
        for raw_pattern in cfg.get("patterns") or []:
            # Pattern can be a regex with one capture group, or a literal prefix.
            pattern = _normalize_regex_pattern(raw_pattern)
            try:
                m = re.search(pattern, haystack, flags=re.IGNORECASE)
            except re.error:
                m = re.search(re.escape(pattern), haystack, flags=re.IGNORECASE)
            if m:
                number = m.group(1) if m.groups() else _extract_number(haystack)
                return {
                    "record_type": record_type,
                    "record_id": number,
                    id_field: number,
                }
    return {"record_type": None, "record_id": None}


def apply_record_identity_policy(
    record: dict[str, Any],
    unit: dict[str, Any] | None,
    profile_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply profile-defined identity rules after LLM/Pydantic validation.

    When override_llm_identity is true, current structural unit hints win over
    the model output. This prevents body cross-references from replacing the
    current record identity.
    """
    policy = _policy(profile_config)
    if not policy.get("enabled", False):
        return record

    identity = infer_record_identity_from_unit(unit, profile_config)
    record_types = policy.get("record_types") or {}
    override = bool(policy.get("override_llm_identity", True))

    record_type = identity.get("record_type")
    record_id = identity.get("record_id")
    if not override or not record_type or not record_id:
        return record

    # Clear all configured id fields first when the policy owns identity.
    for cfg in record_types.values():
        id_field = cfg.get("id_field")
        if id_field:
            record[id_field] = None

    record["record_type"] = record_type
    record["record_id"] = record_id

    cfg = record_types.get(record_type) or {}
    id_field = cfg.get("id_field")
    if id_field:
        record[id_field] = record_id

    return record
