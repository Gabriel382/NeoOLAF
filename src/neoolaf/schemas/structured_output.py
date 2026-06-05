from __future__ import annotations

"""Utilities for optional structured LLM outputs.

The goal of this module is to keep structured-output support configurable and
portable:
- local Pydantic validation can be enabled or disabled by profile;
- LiteLLM response_format can be enabled only when the backend/model supports it;
- normal JSON parsing remains the fallback path.
"""

from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel, Field


class ExtractedItem(BaseModel):
    """Generic text item extracted from one source field."""

    text_en: str | None = None
    text_fr: str | None = None
    evidence_field: str | None = None


class ReferenceItem(BaseModel):
    """Reference to a page, input, diagram, document section, or another record."""

    text_en: str | None = None
    text_fr: str | None = None
    page: str | None = None
    input: str | None = None
    evidence_field: str | None = None


class Layer01AlarmRecord(BaseModel):
    """Layer 1 structured record used by XQuality-like table extraction.

    The name keeps historical compatibility with `alarm_record`, but the schema
    supports both alarms and messages through record_type/message_no.
    """

    record_id: str | None = None
    record_type: str = Field(default="unknown", description="Usually alarm, message, or unknown.")
    alarm_no: str | None = None
    message_no: str | None = None
    alarm_label_en: str | None = None
    alarm_label_fr: str | None = None

    cause_items: list[ExtractedItem] = Field(default_factory=list)
    effect_items: list[ExtractedItem] = Field(default_factory=list)
    intervention_items: list[ExtractedItem] = Field(default_factory=list)
    responsible_items: list[ExtractedItem] = Field(default_factory=list)
    reference_items: list[ReferenceItem] = Field(default_factory=list)


class Layer01AlarmRecordOutput(BaseModel):
    alarm_record: Layer01AlarmRecord


SCHEMA_REGISTRY: dict[str, Type[BaseModel]] = {
    "layer01_alarm_record": Layer01AlarmRecordOutput,
}


@dataclass(frozen=True)
class StructuredOutputConfig:
    enabled: bool = False
    mode: str = "json"
    schema_name: str | None = None
    schema_path: str | None = None
    use_litellm_response_format: bool = False
    fallback_to_json_parse: bool = True
    strict_validation: bool = False

    @classmethod
    def from_profile(cls, profile_config: dict[str, Any] | None, layer_name: str) -> "StructuredOutputConfig":
        profile_config = profile_config or {}
        layer_cfg = (profile_config.get("layers", {}) or {}).get(layer_name, {}) or {}
        raw = layer_cfg.get("structured_output")
        if raw is None:
            # Also support a global mapping for future datasets:
            # {"structured_output": {"layer01...": {...}}}
            global_cfg = profile_config.get("structured_output") or {}
            if isinstance(global_cfg, dict) and layer_name in global_cfg:
                raw = global_cfg[layer_name]
            elif isinstance(global_cfg, dict) and "enabled" in global_cfg:
                raw = global_cfg
        if not isinstance(raw, dict):
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "json")),
            schema_name=raw.get("schema_name"),
            schema_path=raw.get("schema_path"),
            use_litellm_response_format=bool(raw.get("use_litellm_response_format", False)),
            fallback_to_json_parse=bool(raw.get("fallback_to_json_parse", True)),
            strict_validation=bool(raw.get("strict_validation", False)),
        )


def get_schema_model(schema_name: str | None) -> Type[BaseModel] | None:
    if not schema_name:
        return None
    return SCHEMA_REGISTRY.get(schema_name)


def _model_json_schema(model_cls: Type[BaseModel]) -> dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()  # pydantic v2
    return model_cls.schema()  # pydantic v1


def build_litellm_response_format(config: StructuredOutputConfig) -> dict[str, Any] | None:
    """Build a LiteLLM/OpenAI-style response_format dict when configured.

    If the selected backend/model does not support this, the LLM backend wrapper
    should fallback to a normal call when fallback_to_json_parse is true.
    """
    if not config.enabled or not config.use_litellm_response_format:
        return None
    model_cls = get_schema_model(config.schema_name)
    if model_cls is None:
        return None
    schema = _model_json_schema(model_cls)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": config.schema_name or model_cls.__name__,
            "schema": schema,
            "strict": bool(config.strict_validation),
        },
    }


def validate_with_pydantic(
    data: Any,
    config: StructuredOutputConfig,
) -> dict[str, Any]:
    """Validate parsed JSON with a configured Pydantic schema and return a dict."""
    if not config.enabled:
        return data
    model_cls = get_schema_model(config.schema_name)
    if model_cls is None:
        raise ValueError(f"Unknown structured output schema: {config.schema_name!r}")
    if hasattr(model_cls, "model_validate"):
        model = model_cls.model_validate(data)  # pydantic v2
        return model.model_dump()
    model = model_cls.parse_obj(data)  # pydantic v1
    return model.dict()
