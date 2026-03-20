"""Robust JSON extraction from LLM responses.

LLM output often wraps JSON in markdown code blocks or surrounding text.
This module provides a 3-fallback extraction strategy:
1. Direct JSON.loads
2. Regex extraction from ```json``` code blocks
3. Bracket-matching with string-escape awareness
"""

from __future__ import annotations

import json
import re

import structlog

log = structlog.get_logger()


def extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from LLM response text.

    Tries three strategies in order:
    1. Direct parse of the entire text
    2. Regex match of a ```json``` code block
    3. Bracket matching with string-escape handling

    Returns the parsed dict, or None if extraction fails.
    """
    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: code block extraction
    code_match = re.search(r"```(?:json)?\s*(\{.+\})\s*```", text, re.DOTALL)
    if code_match:
        try:
            result = json.loads(code_match.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: bracket matching with string-escape awareness
    return _extract_by_bracket_matching(text)


def _extract_by_bracket_matching(text: str) -> dict | None:
    """Find the first balanced JSON object using bracket counting.

    Handles escaped characters and quoted strings so that braces
    inside string literals don't break the depth tracking.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
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
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None

    return None
