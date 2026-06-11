"""
Inference result validator: validates LLM output before returning to client.
Checks for: empty response, truncated JSON (when JSON mode requested),
repetition loops, excessive whitespace, and null byte contamination.
Logs issues but only blocks on critical failures (null bytes, empty).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
from loguru import logger

_MAX_REPETITION_WINDOW = 50    # chars
_REPETITION_THRESHOLD = 5      # times the same window repeated = loop


@dataclass
class ValidationResult:
    valid: bool
    issues: list[str]
    critical: bool = False    # if True, response should not be sent

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": self.issues,
            "critical": self.critical,
        }


def _detect_repetition_loop(text: str) -> bool:
    """Detects if a window of chars is repeated >= THRESHOLD times consecutively."""
    if len(text) < _MAX_REPETITION_WINDOW * _REPETITION_THRESHOLD:
        return False
    # Check last portion of text for repetition
    tail = text[-_MAX_REPETITION_WINDOW * _REPETITION_THRESHOLD:]
    window = tail[:_MAX_REPETITION_WINDOW]
    escaped = re.escape(window)
    matches = re.findall(escaped, tail)
    return len(matches) >= _REPETITION_THRESHOLD


class InferenceValidator:
    """
    validate(text, json_mode): checks response text for known failure modes.
    Returns ValidationResult — critical=True means caller should return 500.
    """

    def __init__(self) -> None:
        self._total_validated = 0
        self._total_issues = 0
        self._total_critical = 0

    def validate(self, text: str, json_mode: bool = False) -> ValidationResult:
        self._total_validated += 1
        issues: list[str] = []
        critical = False

        # Critical: empty response
        if not text or not text.strip():
            issues.append("empty_response")
            critical = True

        # Critical: null bytes
        if "\x00" in text:
            issues.append("null_byte_contamination")
            critical = True
            text = text.replace("\x00", "")

        # Non-critical: truncated JSON
        if json_mode and text.strip():
            stripped = text.strip()
            if stripped.startswith("{") and not stripped.endswith("}"):
                issues.append("truncated_json")
            elif stripped.startswith("[") and not stripped.endswith("]"):
                issues.append("truncated_json_array")

        # Non-critical: repetition loop
        if text and _detect_repetition_loop(text):
            issues.append("repetition_loop")

        # Non-critical: excessive whitespace (>50% of content)
        if text:
            whitespace_ratio = sum(1 for c in text if c in " \t\n\r") / len(text)
            if whitespace_ratio > 0.5:
                issues.append("excessive_whitespace")

        if issues:
            self._total_issues += 1
            level = "critical" if critical else "warning"
            logger.log(
                level.upper(),
                f"InferenceValidator: issues={issues} critical={critical} "
                f"text_len={len(text)}"
            )
        if critical:
            self._total_critical += 1

        return ValidationResult(valid=not critical, issues=issues, critical=critical)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_validated": self._total_validated,
            "total_with_issues": self._total_issues,
            "total_critical": self._total_critical,
        }


INFERENCE_VALIDATOR = InferenceValidator()
