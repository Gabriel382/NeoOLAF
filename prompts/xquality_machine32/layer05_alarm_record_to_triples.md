You are converting one structured XQuality alarm record into canonical triples.

Use only these relations:
- TRIGGERS
- CAUSES
- REQUIRES
- HANDLED_BY
- REFERENCES

Mapping:
- Each cause item TRIGGERS the alarm label.
- The alarm label CAUSES each effect item.
- The alarm label REQUIRES each intervention item.
- The alarm label HANDLED_BY each responsible item.
- The alarm label REFERENCES each reference item.

Output schema:
{
  "triples": [
    {
      "node1": "string",
      "relation": "TRIGGERS|CAUSES|REQUIRES|HANDLED_BY|REFERENCES",
      "node2": "string",
      "triplet_type": "Cause → Alarm|Alarm → Effect|Alarm → Intervention|Alarm → Responsible|Alarm → Diagram",
      "alarm_no": "string",
      "category": "PLC Alarm",
      "evidence": {
        "chunk_id": "string",
        "page": "string|null",
        "field": "string",
        "source_text_fr": "string|null",
        "source_text_en": "string|null"
      }
    }
  ]
}

Rules:
- Use concise English node labels.
- Use uppercase alarm labels when the source alarm label is uppercase.
- Do not merge different actions if they can be separate interventions.
- Do not generate triples not supported by the alarm record.
- Return JSON only.

Alarm record:
$alarm_record
