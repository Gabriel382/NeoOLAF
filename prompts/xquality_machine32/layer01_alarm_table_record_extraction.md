You are processing one industrial alarm table from an XQuality Machine 32 CNC manual.

Task:
Extract one structured alarm record from the provided table chunk.
Do not generate triples yet.
Use only the provided table/chunk text and its metadata.
Translate extracted French content to concise English.
Return valid JSON only.

Allowed JSON schema:
{
  "alarm_record": {
    "alarm_no": "string",
    "alarm_label_en": "string",
    "alarm_label_fr": "string",
    "cause_items": [
      {"text_en": "string", "text_fr": "string", "evidence_field": "cause"}
    ],
    "effect_items": [
      {"text_en": "string", "text_fr": "string", "evidence_field": "effect"}
    ],
    "intervention_items": [
      {"text_en": "string", "text_fr": "string", "evidence_field": "intervention"}
    ],
    "responsible_items": [
      {"text_en": "string", "text_fr": "string", "evidence_field": "responsible"}
    ],
    "reference_items": [
      {"text_en": "string", "text_fr": "string", "page": "string|null", "input": "string|null", "evidence_field": "reference|intervention|other"}
    ]
  }
}

Rules:
- The alarm label is the value of the "texte"/"text" field, usually uppercase.
- Split multi-action cells into separate intervention_items when they contain distinct actions.
- If a page/input/schema reference appears inside an intervention cell, put it in reference_items, not intervention_items.
- Preserve technical identifiers exactly: X8.4, M560, A2831, M08, etc.
- Keep alarm labels uppercase if the source label is uppercase.
- Do not infer missing information.
- Do not include explanations outside JSON.

Chunk metadata:
$chunk_metadata

Table/chunk text:
$chunk_text
