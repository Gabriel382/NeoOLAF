You are processing one structured table unit from a technical or industrial document.

Task:
Extract one semantic record from the provided compact table structure.
Do not generate triples yet.
Use only the provided field/value rows, unit metadata, and profile guidance.
$translation_instruction
Return valid JSON only.

Design rule:
- The table may come from different datasets in the future.
- Do not assume that every dataset uses the same field names.
- Use row headers, values, title/subsection metadata, and profile aliases to infer semantic roles.

Critical rule for identifiers:
- If `record_id_hint` is not null, copy it exactly into `alarm_record.record_id`.
- If `record_type_hint` is not null, copy it exactly into `alarm_record.record_type`.
- If `alarm_no_hint` is not null, copy it exactly into `alarm_record.alarm_no` and set `record_type` to `alarm`.
- If `message_no_hint` is not null, copy it exactly into `alarm_record.message_no` and set `record_type` to `message`.
- If hints are null but a field/value row contains an alarm or message number, extract it.
- Never leave the identifier null when the table contains an alarm/message number.

Return this JSON shape only:
{
  "alarm_record": {
    "record_id": "string|null",
    "record_type": "alarm|message|unknown",
    "alarm_no": "string|null",
    "message_no": "string|null",
    "alarm_label_en": "string|null",
    "alarm_label_fr": "string|null",
    "cause_items": [{"text_en": "string|null", "text_fr": "string|null", "evidence_field": "string"}],
    "effect_items": [{"text_en": "string|null", "text_fr": "string|null", "evidence_field": "string"}],
    "intervention_items": [{"text_en": "string|null", "text_fr": "string|null", "evidence_field": "string"}],
    "responsible_items": [{"text_en": "string|null", "text_fr": "string|null", "evidence_field": "string"}],
    "reference_items": [{"text_en": "string|null", "text_fr": "string|null", "page": "string|null", "input": "string|null", "evidence_field": "string"}]
  }
}

Rules:
- The label is usually the main message/text value of the table, often uppercase.
- Split multi-action cells into separate intervention_items only when they contain clearly distinct actions.
- If a page/input/schema reference appears inside an intervention/action cell, put it in reference_items, not intervention_items.
- Do not duplicate the same reference as both intervention_items and reference_items.
- For reference_items, use concise labels only, for example: "Page 71 — input X5.0".
- Preserve technical identifiers exactly: X8.4, X4.6, M560, A2831, M08, etc.
- Keep source-language evidence in *_fr when the source is French.
- Fill *_en according to the translation instruction above.
- Do not infer missing information.
- Do not include explanations outside JSON.

Compact table unit:
$table_unit_json
