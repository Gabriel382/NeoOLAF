"""Constants and default paths for the XQuality TaxoDrivenKG adaptation."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PATH = {
    "inputs": {
        "state_json": str(BASE_DIR / "inputs" / "state.json"),
        "ontology": str(BASE_DIR / "inputs" / "ontology.owl"),
        "env_file": str(BASE_DIR.parent.parent.parent / ".env"),
    },
    "outputs": {
        "base": str(BASE_DIR / "outputs"),
        "prompts": str(BASE_DIR / "outputs_prompts"),
        "conversations": str(BASE_DIR / "conversations"),
        "ttl": str(BASE_DIR / "outputs_ttl"),
    },
}

LABELS_DICT = {
    "entities": [
        "machine", "component", "alarm", "message", "event", "failure", "fault",
        "cause", "effect", "intervention", "operator", "maintenance role",
        "sensor", "signal", "parameter", "procedure", "document", "location",
    ],
    "relations": [
        "causes", "affects", "indicates", "requires", "performedBy", "refersTo",
        "hasIntervention", "hasEffect", "hasCause", "observedAt", "partOf",
        "connectedTo", "triggers", "prevents",
    ],
}

text_template = "<heading>{}</heading>\n{}\n"
