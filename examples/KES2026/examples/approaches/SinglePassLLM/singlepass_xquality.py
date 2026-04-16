from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================================================
# Basic I/O
# =========================================================
def load_json_file(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str | Path, data: Any) -> None:
    """Save a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_text_file(path: str | Path) -> str:
    """Load a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text_file(path: str | Path, text: Any) -> None:
    """
    Save a text file safely even if text is None or another type.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================================================
# NeoOLAF preprocessing loader
# =========================================================
def load_translated_text_from_state(state_json_path: str | Path) -> Tuple[str, List[dict]]:
    """
    Load NeoOLAF layer00 state JSON and concatenate only translated_text blocks.
    """
    state = load_json_file(state_json_path)
    blocks = state.get("content_blocks", [])

    translated_blocks: List[dict] = []
    merged_parts: List[str] = []

    for block in blocks:
        translated_text = (block.get("translated_text") or "").strip()
        if translated_text:
            translated_blocks.append(block)
            merged_parts.append(translated_text)

    full_text = "\n\n".join(merged_parts)
    return full_text, translated_blocks


# =========================================================
# Ontology loader
# =========================================================
def load_ontology_snippet(ontology_path: str | Path, max_chars: int = 20000) -> str:
    """
    Load a truncated ontology snippet as plain text.
    """
    if ontology_path is None:
        return ""

    text = load_text_file(ontology_path)
    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...\n[TRUNCATED]"


# =========================================================
# Backend builders
# =========================================================
def build_backend(
    backend_name: str,
    host: str,
    api_key: str = "dummy",
    referer: str = "",
    title: str = "SinglePass-XQuality",
):
    """
    Build a backend wrapper or config dict.
    """
    backend_name = backend_name.lower()

    if backend_name == "ollama":
        return {
            "backend_name": "ollama",
            "host": host.rstrip("/"),
        }

    if backend_name in {"vllm", "openrouter"}:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if backend_name == "openrouter":
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title

        return {
            "backend_name": backend_name,
            "host": host.rstrip("/"),
            "headers": headers,
        }

    raise ValueError(f"Unsupported backend_name: {backend_name}")


# =========================================================
# Prompting
# =========================================================
def build_singlepass_prompt(document_text: str, ontology_text: str = "") -> str:
    """
    Build a strict single-pass KG extraction prompt.
    """
    ontology_block = ""
    if ontology_text.strip():
        ontology_block = f"""
SEED ONTOLOGY (optional guidance only):
--------------------
{ontology_text}
--------------------
"""

    return f"""
You are an information extraction system for industrial knowledge graph construction.

Your task:
Given an industrial document, produce a JSON object describing a knowledge graph extraction in a SINGLE PASS.
Do not perform retrieval, no multi-step reasoning, and no ontology evolution. Just extract a final KG proposal.

Return ONLY valid JSON. No markdown fences. No explanations before or after JSON.

Expected JSON format:
{{
  "entities": [
    {{
      "id": "E1",
      "label": "entity label",
      "type": "entity type or best guess"
    }}
  ],
  "relations": [
    {{
      "head": "entity label exactly as written in entities",
      "relation": "TRIGGERS|CAUSES|REQUIRES|HANDLED_BY|REFERENCES",
      "tail": "entity label exactly as written in entities",
      "evidence": "short supporting text span from the document"
    }}
  ]
}}

Rules:
- Use only these relation labels exactly:
  - TRIGGERS
  - CAUSES
  - REQUIRES
  - HANDLED_BY
  - REFERENCES
- Prefer concrete alarm, cause, intervention, responsible actor, or referenced artifact entities.
- Keep entity labels concise but faithful to the document.
- Each relation must use head/tail labels that appear in "entities".
- "evidence" must be copied or closely grounded in the document.
- If unsure, omit rather than hallucinate.
- Output must be strict JSON parseable by json.loads.

{ontology_block}

DOCUMENT:
--------------------
{document_text}
--------------------
""".strip()


# =========================================================
# JSON extraction helpers
# =========================================================
def strip_code_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()

    import re

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    return text


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Try to extract the first top-level JSON object from a messy response.
    """
    if not text:
        return None

    text = strip_code_fences(text)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def normalize_parsed_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure parsed output has the expected top-level keys.
    """
    entities = data.get("entities", [])
    relations = data.get("relations", [])

    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    return {
        "entities": entities,
        "relations": relations,
    }


# =========================================================
# Backend call
# =========================================================
def call_backend_generate(
    backend: Dict[str, Any],
    backend_name: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    timeout: Optional[int] = None,
) -> str:
    """
    Call backend and always return a string.
    """
    backend_name = backend_name.lower()

    if backend_name in {"vllm", "openrouter"}:
        url = backend["host"] + "/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a precise industrial KG extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            url,
            headers=backend["headers"],
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            return json.dumps(data, ensure_ascii=False, indent=2)

        message = choices[0].get("message", {})
        content = message.get("content")

        if content is None:
            return json.dumps(data, ensure_ascii=False, indent=2)

        if isinstance(content, list):
            # Some APIs may return structured content parts.
            parts = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if txt:
                        parts.append(txt)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()

        return str(content).strip()

    if backend_name == "ollama":
        url = backend["host"] + "/api/chat"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a precise industrial KG extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        content = message.get("content")

        if content is None:
            return json.dumps(data, ensure_ascii=False, indent=2)

        return str(content).strip()

    raise ValueError(f"Unsupported backend_name: {backend_name}")


# =========================================================
# Main runner
# =========================================================
def run_singlepass_xquality(
    state_json_path: str | Path,
    ontology_path: str | Path | None,
    backend_name: str,
    host: str,
    model_name: str,
    api_key: str = "dummy",
    use_ontology: bool = True,
    ontology_max_chars: int = 20000,
    referer: str = "",
    title: str = "SinglePass-XQuality",
    output_dir: str | Path = "./outputs_singlepass_xquality",
    output_stem: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the single-pass XQuality baseline and save prompt/raw/parsed outputs.
    """
    # Load text from NeoOLAF state.
    document_text, translated_blocks = load_translated_text_from_state(state_json_path)

    ontology_text = ""
    if use_ontology and ontology_path is not None:
        ontology_text = load_ontology_snippet(ontology_path, max_chars=ontology_max_chars)

    prompt = build_singlepass_prompt(
        document_text=document_text,
        ontology_text=ontology_text,
    )

    backend = build_backend(
        backend_name=backend_name,
        host=host,
        api_key=api_key,
        referer=referer,
        title=title,
    )

    raw_response = call_backend_generate(
        backend=backend,
        backend_name=backend_name,
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_stem is None:
        output_stem = Path(state_json_path).stem

    prompt_path = output_dir / f"{output_stem}__singlepass_prompt.txt"
    raw_path = output_dir / f"{output_stem}__singlepass_raw.txt"
    parsed_path = output_dir / f"{output_stem}__singlepass_parsed.json"

    save_text_file(prompt_path, prompt)
    save_text_file(raw_path, raw_response)

    parsed_ok = False
    parsing_error = None
    parsed_data = None
    final_parsed_path = None

    try:
        extracted_json = extract_first_json_object(raw_response)
        if extracted_json is None:
            raise ValueError("No JSON object found in model response.")

        parsed_data = json.loads(extracted_json)
        parsed_data = normalize_parsed_output(parsed_data)

        save_json_file(parsed_path, parsed_data)
        parsed_ok = True
        final_parsed_path = str(parsed_path)

    except Exception as e:
        parsing_error = str(e)

    return {
        "parsed_ok": parsed_ok,
        "parsing_error": parsing_error,
        "prompt_path": str(prompt_path),
        "raw_path": str(raw_path),
        "parsed_path": final_parsed_path,
        "raw_response": raw_response,
        "parsed_data": parsed_data,
        "translated_blocks_count": len(translated_blocks),
        "document_chars": len(document_text),
    }