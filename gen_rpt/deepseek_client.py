from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests


class DeepSeekClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout: int = 180,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "Missing DEEPSEEK_API_KEY. Please configure it in GitHub Actions Secrets or your local environment."
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        model: Optional[str] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw = self.chat(messages, temperature=temperature, model=model)
        try:
            return extract_json_object(raw)
        except Exception as first_error:
            repaired_locally = repair_json_like(raw)
            try:
                return extract_json_object(repaired_locally)
            except Exception:
                pass

            # Keep the repair payload bounded. Sending a 35k+ invalid JSON blob back
            # can cause the model to echo the same syntax error. Around the parse
            # location is usually enough for the model, but we still include the full
            # object when it is reasonably small.
            repair_source = repaired_locally if len(repaired_locally) <= 24000 else _compact_for_repair(repaired_locally)
            repair_messages = [
                {
                    "role": "system",
                    "content": "You repair invalid JSON. Return valid JSON only. Do not add markdown or commentary.",
                },
                {
                    "role": "user",
                    "content": (
                        "The following model output was intended to be one JSON object, but it is invalid. "
                        "Repair JSON syntax only. Preserve all available keys, text, numbers, arrays and objects. "
                        "Most likely issues are missing commas between array objects or object properties. "
                        "Return one valid JSON object only.\n\n"
                        f"Parse error: {first_error}\n\n"
                        f"Invalid JSON-like output:\n{repair_source}"
                    ),
                },
            ]
            repaired = self.chat(repair_messages, temperature=0.0, model=model)
            try:
                return extract_json_object(repaired)
            except Exception as second_error:
                # Last local pass on the model's repaired response.
                try:
                    return extract_json_object(repair_json_like(repaired))
                except Exception as third_error:
                    raise ValueError(
                        "DeepSeek returned invalid JSON and automatic repair failed. "
                        f"Initial parse error: {first_error}. Repair parse error: {second_error}. "
                        f"Final local repair error: {third_error}. Raw response excerpt: {raw[:1200]}"
                    ) from third_error


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(str(text or "").strip())
    cleaned = _extract_json_like(cleaned)
    cleaned = repair_json_like(cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        snippet = _error_snippet(cleaned, exc.pos)
        raise json.JSONDecodeError(f"{exc.msg}. Nearby text: {snippet}", exc.doc, exc.pos) from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


def repair_json_like(text: str) -> str:
    """Repair common LLM JSON syntax mistakes without changing content.

    This is deliberately conservative but handles the failure pattern we see most:
    objects in arrays emitted as `}{` instead of `},{`, or finished arrays/objects
    followed by the next quoted key without a comma.
    """
    fixed = _strip_code_fences(str(text or "").strip())
    fixed = _extract_json_like(fixed)
    fixed = fixed.replace("\ufeff", "")
    fixed = fixed.replace("\u0000", "")

    # Normalize smart quotes only when they are used as structural quotes. Content
    # quotes inside already-valid strings are left untouched by the JSON parser.
    fixed = fixed.replace("“", '"').replace("”", '"')

    # Missing comma between adjacent objects, e.g. in references: `}\n    {`.
    fixed = re.sub(r"}\s*{", "},\n{", fixed)

    # Missing comma between a finished array/object/string/number and the next key.
    # Examples: `]\n \"charts\":`, `}\n \"references\":`, `\"x\"\n \"y\":`.
    fixed = re.sub(r"([}\]])\s*(\"[A-Za-z_][A-Za-z0-9_\-]*\"\s*:)", r"\1,\n\2", fixed)
    fixed = re.sub(r"(\"(?:[^\"\\]|\\.)*\")\s*(\"[A-Za-z_][A-Za-z0-9_\-]*\"\s*:)", r"\1,\n\2", fixed)
    fixed = re.sub(r"(-?\d+(?:\.\d+)?)\s*(\"[A-Za-z_][A-Za-z0-9_\-]*\"\s*:)", r"\1,\n\2", fixed)

    # Missing comma between string array items: `"a"\n "b"`.
    fixed = re.sub(r"(\"(?:[^\"\\]|\\.)*\")\s*(\"(?:[^\"\\]|\\.)*\")(?=\s*[,\]])", r"\1,\n\2", fixed)

    # Remove trailing commas after the repairs.
    fixed = _remove_trailing_commas(fixed)
    return fixed


def _strip_code_fences(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text


def _extract_json_like(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return a JSON object. Raw response excerpt:\n{text[:1200]}")
    return text[start : end + 1]


def _remove_trailing_commas(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = re.sub(r",\s*([}\]])", r"\1", current)
    return current


def _compact_for_repair(text: str, max_chars: int = 24000) -> str:
    if len(text) <= max_chars:
        return text
    # Keep the beginning and end. In our failures, the invalid references array is
    # typically near the end; the beginning preserves schema context.
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n\n/* middle omitted only for syntax repair */\n\n" + tail


def _error_snippet(text: str, pos: int, radius: int = 240) -> str:
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return text[start:end].replace("\n", " ")
