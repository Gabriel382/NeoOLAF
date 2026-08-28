from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from olaf.commons.llm_tools import LLMGenerator


def _balanced_list_candidates(text: str) -> list[str]:
    candidates = []
    starts = []
    in_string = False
    quote = None
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
                quote = None
            continue

        if ch in {"'", '"'}:
            in_string = True
            quote = ch
        elif ch == "[":
            starts.append(i)
        elif ch == "]" and starts:
            start = starts.pop()
            candidates.append(text[start:i + 1])

    return sorted(candidates, key=len, reverse=True)


def _coerce_groups(obj: Any) -> list[list[str]] | None:
    if isinstance(obj, dict):
        for key in ("groups", "concepts", "relations", "result", "output", "data"):
            if key in obj:
                return _coerce_groups(obj[key])
        return None

    if not isinstance(obj, list):
        return None

    if all(isinstance(x, str) for x in obj):
        return [[x] for x in obj]

    groups: list[list[str]] = []
    for group in obj:
        if isinstance(group, str):
            groups.append([group])
        elif isinstance(group, (list, tuple, set)):
            strings = [x for x in group if isinstance(x, str)]
            if strings:
                groups.append(strings)
        elif isinstance(group, dict):
            for key in ("terms", "words", "items", "members"):
                if key in group and isinstance(group[key], list):
                    strings = [x for x in group[key] if isinstance(x, str)]
                    if strings:
                        groups.append(strings)
                    break

    return groups if groups else []


def _norm_key(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _extract_allowed_labels(messages: List[Dict[str, str]]) -> list[str]:
    """Extract OLAF's candidate vocabulary from its own prompt.

    OLAF's native prompt puts one candidate term per line after `Words :`.
    No dataset gold, ontology labels, or benchmark information is used here.
    """
    for message in reversed(messages):
        content = str(message.get("content", ""))
        if "Words :" not in content:
            continue
        words_block = content.rsplit("Words :", 1)[1]
        labels = [line.strip() for line in words_block.splitlines() if line.strip()]
        # preserve prompt order while deduplicating
        return list(dict.fromkeys(labels))
    return []


def _ground_groups_to_allowed(
    groups: list[list[str]], allowed_labels: Iterable[str]
) -> tuple[list[list[str]], list[str]]:
    """Keep only OLAF candidate labels.

    GPT models occasionally rename/paraphrase a term even when told not to.
    OLAF then creates an empty candidate group and crashes in cts_to_concept()
    at candidates[0]. This guard enforces OLAF's own contract: group only the
    candidate strings supplied in the prompt.

    Exact matches are preferred. A whitespace/case-normalized match is accepted
    only when it maps uniquely to one original candidate label.
    """
    allowed = list(dict.fromkeys(str(x) for x in allowed_labels))
    exact = set(allowed)

    normalized_map: dict[str, list[str]] = {}
    for label in allowed:
        normalized_map.setdefault(_norm_key(label), []).append(label)

    grounded: list[list[str]] = []
    dropped: list[str] = []
    seen_global: set[str] = set()

    for group in groups:
        clean_group: list[str] = []

        for item in group:
            resolved = None
            if item in exact:
                resolved = item
            else:
                matches = normalized_map.get(_norm_key(item), [])
                if len(matches) == 1:
                    resolved = matches[0]

            if resolved is None:
                dropped.append(item)
                continue

            # A candidate should belong to at most one OLAF grouping.
            if resolved in seen_global:
                continue

            seen_global.add(resolved)
            clean_group.append(resolved)

        # Critical OLAF compatibility rule: NEVER return an empty group.
        if clean_group:
            grounded.append(clean_group)

    return grounded, dropped


def normalize_olaf_grouping_output(
    raw: str,
    allowed_labels: Iterable[str] | None = None,
) -> tuple[str, list[str]]:
    """Return OLAF-compatible `list[list[str]]` plus dropped unknown labels."""
    if raw is None:
        raise RuntimeError("OpenRouter returned no message content.")

    text = str(raw).strip()
    if not text:
        raise RuntimeError("OpenRouter returned empty message content.")

    text = re.sub(r"^\s*```(?:json|python|py)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)

    attempts = [text] + _balanced_list_candidates(text)

    for candidate in attempts:
        candidate = candidate.strip()

        for parser in (ast.literal_eval, json.loads):
            try:
                obj = parser(candidate)
            except Exception:
                continue

            groups = _coerce_groups(obj)
            if groups is None:
                continue

            dropped: list[str] = []
            if allowed_labels is not None:
                groups, dropped = _ground_groups_to_allowed(groups, allowed_labels)

            # repr is deliberate: OLAF itself calls ast.literal_eval().
            return repr(groups), dropped

    raise RuntimeError(
        "GPT-OSS response could not be normalized to OLAF's required "
        "list-of-lists format. Raw response preview:\n" + text[:1200]
    )


class OpenRouterGenerator(LLMGenerator):
    """OLAF LLMGenerator backed by OpenRouter.

    Adaptation scope:
      - OpenAI-compatible OpenRouter transport
      - output syntax normalization
      - enforcement of OLAF's own candidate-term vocabulary
    No benchmark labels or gold annotations are supplied to the model.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 2048,
        reasoning_effort: str = "minimal",
        debug_log_path: str | Path | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "OLAF_OPENROUTER_MODEL", "openai/gpt-oss-20b"
        )
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.debug_log_path = Path(debug_log_path) if debug_log_path else None
        self.call_index = 0

    def check_resources(self) -> None:
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("OPENROUTER_API_KEY is missing.")

    def _client(self) -> OpenAI:
        self.check_resources()
        return OpenAI(
            base_url=self.base_url,
            api_key=os.environ["OPENROUTER_API_KEY"].strip(),
            timeout=180.0,
        )

    def _prepare_messages(self, prompt: Any) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        else:
            messages = [dict(m) for m in prompt]

        messages.append(
            {
                "role": "user",
                "content": (
                    "Formatting constraint only: group ONLY the exact candidate "
                    "strings supplied after `Words :`. Do not invent, rename, "
                    "lemmatize, paraphrase, or normalize any candidate. Return ONLY "
                    "a top-level array of arrays of strings. Omit a group rather than "
                    "returning a term that is not in the supplied candidate list. "
                    "No markdown, explanation, comments, or object wrapper."
                ),
            }
        )
        return messages

    def _prompt_hash(
        self, messages: List[Dict[str, str]], allowed_labels: list[str]
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "allowed_labels": allowed_labels,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _log(
        self,
        *,
        prompt_hash: str,
        allowed_labels: list[str],
        raw: str,
        normalized: str | None,
        dropped_unknown_labels: list[str],
        usage: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if self.debug_log_path is None:
            return
        self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "call_index": self.call_index,
            "model": self.model_name,
            "prompt_hash": prompt_hash,
            "candidate_count": len(allowed_labels),
            "raw": raw,
            "normalized": normalized,
            "dropped_unknown_labels": dropped_unknown_labels,
            "usage": usage or {},
            "error": error,
        }
        with self.debug_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @retry(
        retry=retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate_text(self, prompt: Any) -> str:
        self.call_index += 1

        # Extract candidate vocabulary BEFORE appending our format-only message.
        base_messages: List[Dict[str, str]]
        if isinstance(prompt, str):
            base_messages = [{"role": "user", "content": prompt}]
        else:
            base_messages = [dict(m) for m in prompt]

        allowed_labels = _extract_allowed_labels(base_messages)
        messages = self._prepare_messages(prompt)
        prompt_hash = self._prompt_hash(messages, allowed_labels)

        response = self._client().chat.completions.create(
            model=self.model_name,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=messages,
            extra_body={
                "reasoning": {
                    "effort": self.reasoning_effort,
                    "exclude": True,
                },
                # OpenRouter returns provider token accounting in the response.
                "usage": {"include": True},
            },
        )

        raw = response.choices[0].message.content or ""

        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            usage = {}
        elif hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage = dict(usage_obj)
        else:
            usage = {}

        # Keep only JSON-safe token accounting fields. OpenRouter's standard
        # fields are prompt_tokens, completion_tokens and total_tokens.
        usage = {
            k: v for k, v in usage.items()
            if isinstance(v, (int, float, str, bool, type(None), dict, list))
        }

        try:
            normalized, dropped = normalize_olaf_grouping_output(
                raw,
                allowed_labels=allowed_labels if allowed_labels else None,
            )
        except Exception as exc:
            self._log(
                prompt_hash=prompt_hash,
                allowed_labels=allowed_labels,
                raw=raw,
                normalized=None,
                dropped_unknown_labels=[],
                usage=usage,
                error=str(exc),
            )
            raise

        self._log(
            prompt_hash=prompt_hash,
            allowed_labels=allowed_labels,
            raw=raw,
            normalized=normalized,
            dropped_unknown_labels=dropped,
            usage=usage,
            error=None,
        )
        return normalized
