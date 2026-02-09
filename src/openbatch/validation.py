"""
Validation utilities for OpenAI batch job files.

This module provides functions to validate JSONL batch files before uploading them
to the OpenAI Batch API, helping catch errors early and ensure compliance with
API requirements.
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any, Set
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """
    Result of batch file validation.

    Attributes:
        is_valid (bool): Whether the batch file is valid
        errors (List[str]): List of validation errors found
        warnings (List[str]): List of non-critical warnings
        stats (Dict[str, Any]): Statistics about the batch file
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable summary of validation results."""
        lines = []
        lines.append(f"Validation: {'PASSED' if self.is_valid else 'FAILED'}")

        if self.stats:
            lines.append("\nStatistics:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  • {error}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  • {warning}")

        return "\n".join(lines)


class BatchFileValidator:
    """
    Validator for OpenAI batch job JSONL files.

    Validates batch files against OpenAI Batch API requirements including:
    - Valid JSONL format
    - Unique custom_ids
    - Required fields present
    - Valid endpoint URLs
    - File size limits
    """

    # OpenAI Batch API constraints (as of 2026)
    MAX_FILE_SIZE_MB = 200
    MAX_REQUESTS = 50000
    VALID_ENDPOINTS = {
        "/v1/responses",
        "/v1/chat/completions",
        "/v1/embeddings"
    }
    REQUIRED_FIELDS = {"custom_id", "method", "url", "body"}

    def __init__(
        self,
        check_custom_id_uniqueness: bool = True,
        check_file_size: bool = True,
        check_request_count: bool = True,
        allow_mixed_endpoints: bool = False
    ):
        """
        Initialize the validator with configuration options.

        Args:
            check_custom_id_uniqueness: Check for duplicate custom_ids
            check_file_size: Check file size against limits
            check_request_count: Check number of requests against limits
            allow_mixed_endpoints: Allow multiple endpoint types in one file (not recommended)
        """
        self.check_custom_id_uniqueness = check_custom_id_uniqueness
        self.check_file_size = check_file_size
        self.check_request_count = check_request_count
        self.allow_mixed_endpoints = allow_mixed_endpoints

    def validate_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a batch file.

        Args:
            file_path: Path to the JSONL batch file

        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        file_path = Path(file_path)
        result = ValidationResult(is_valid=True)

        # Check file exists
        if not file_path.exists():
            result.errors.append(f"File not found: {file_path}")
            result.is_valid = False
            return result

        # Check file extension
        if file_path.suffix != ".jsonl":
            result.warnings.append(
                f"File extension is '{file_path.suffix}', expected '.jsonl'"
            )

        # Check file size
        if self.check_file_size:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            result.stats["file_size_mb"] = round(file_size_mb, 2)

            if file_size_mb > self.MAX_FILE_SIZE_MB:
                result.errors.append(
                    f"File size ({file_size_mb:.2f} MB) exceeds limit "
                    f"({self.MAX_FILE_SIZE_MB} MB)"
                )
                result.is_valid = False

        # Validate content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self._validate_content(f, result)
        except Exception as e:
            result.errors.append(f"Error reading file: {str(e)}")
            result.is_valid = False

        return result

    def _validate_content(self, file_handle, result: ValidationResult) -> None:
        """Validate the content of the batch file."""
        custom_ids: Set[str] = set()
        endpoints: Set[str] = set()
        line_number = 0

        for line in file_handle:
            line_number += 1
            line = line.strip()

            # Skip empty lines
            if not line:
                result.warnings.append(f"Line {line_number}: Empty line (will be ignored)")
                continue

            # Parse JSON
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                result.errors.append(
                    f"Line {line_number}: Invalid JSON - {str(e)}"
                )
                result.is_valid = False
                continue

            # Validate request structure
            self._validate_request(request, line_number, custom_ids, endpoints, result)

        # Update statistics
        result.stats["total_requests"] = line_number
        result.stats["unique_custom_ids"] = len(custom_ids)
        result.stats["endpoints_used"] = list(endpoints)

        # Check request count
        if self.check_request_count and line_number > self.MAX_REQUESTS:
            result.errors.append(
                f"Request count ({line_number}) exceeds limit ({self.MAX_REQUESTS})"
            )
            result.is_valid = False

        # Check for mixed endpoints
        if not self.allow_mixed_endpoints and len(endpoints) > 1:
            result.warnings.append(
                f"Multiple endpoint types detected: {list(endpoints)}. "
                "OpenAI recommends one request type per file."
            )

    def _validate_request(
        self,
        request: Dict[str, Any],
        line_number: int,
        custom_ids: Set[str],
        endpoints: Set[str],
        result: ValidationResult
    ) -> None:
        """Validate a single request object."""

        # Check required fields
        missing_fields = self.REQUIRED_FIELDS - set(request.keys())
        if missing_fields:
            result.errors.append(
                f"Line {line_number}: Missing required fields: {missing_fields}"
            )
            result.is_valid = False
            return

        # Validate custom_id
        custom_id = request.get("custom_id")
        if not custom_id or not isinstance(custom_id, str):
            result.errors.append(
                f"Line {line_number}: Invalid custom_id (must be a non-empty string)"
            )
            result.is_valid = False
        elif self.check_custom_id_uniqueness:
            if custom_id in custom_ids:
                result.errors.append(
                    f"Line {line_number}: Duplicate custom_id '{custom_id}'"
                )
                result.is_valid = False
            else:
                custom_ids.add(custom_id)

        # Validate method
        method = request.get("method")
        if method != "POST":
            result.errors.append(
                f"Line {line_number}: Invalid method '{method}' (must be 'POST')"
            )
            result.is_valid = False

        # Validate URL
        url = request.get("url")
        if url not in self.VALID_ENDPOINTS:
            result.errors.append(
                f"Line {line_number}: Invalid endpoint '{url}'. "
                f"Valid endpoints: {self.VALID_ENDPOINTS}"
            )
            result.is_valid = False
        else:
            endpoints.add(url)

        # Validate body
        body = request.get("body")
        if not isinstance(body, dict):
            result.errors.append(
                f"Line {line_number}: 'body' must be a JSON object"
            )
            result.is_valid = False
        else:
            self._validate_body(body, url, line_number, result)

    def _validate_body(
        self,
        body: Dict[str, Any],
        endpoint: str,
        line_number: int,
        result: ValidationResult
    ) -> None:
        """Validate the request body based on endpoint type."""

        # Check for model field (required for all endpoints)
        if "model" not in body:
            result.errors.append(
                f"Line {line_number}: Missing required field 'model' in body"
            )
            result.is_valid = False

        # Endpoint-specific validation
        if endpoint == "/v1/responses":
            if "input" not in body and "prompt" not in body:
                result.errors.append(
                    f"Line {line_number}: Responses API requires either 'input' or 'prompt' in body"
                )
                result.is_valid = False

        elif endpoint == "/v1/chat/completions":
            if "messages" not in body:
                result.errors.append(
                    f"Line {line_number}: Chat Completions API requires 'messages' in body"
                )
                result.is_valid = False
            elif not isinstance(body["messages"], list):
                result.errors.append(
                    f"Line {line_number}: 'messages' must be an array"
                )
                result.is_valid = False

        elif endpoint == "/v1/embeddings":
            if "input" not in body:
                result.errors.append(
                    f"Line {line_number}: Embeddings API requires 'input' in body"
                )
                result.is_valid = False


def validate_batch_file(
    file_path: Union[str, Path],
    strict: bool = True,
    check_custom_id_uniqueness: bool = True,
    allow_mixed_endpoints: bool = False
) -> ValidationResult:
    """
    Validate a batch file (convenience function).

    Args:
        file_path: Path to the JSONL batch file
        strict: Enable all checks (file size, request count)
        check_custom_id_uniqueness: Check for duplicate custom_ids
        allow_mixed_endpoints: Allow multiple endpoint types in one file

    Returns:
        ValidationResult with errors, warnings, and statistics

    Example:
        >>> result = validate_batch_file("my_batch.jsonl")
        >>> if result.is_valid:
        ...     print("File is valid!")
        ... else:
        ...     print(result)
    """
    validator = BatchFileValidator(
        check_custom_id_uniqueness=check_custom_id_uniqueness,
        check_file_size=strict,
        check_request_count=strict,
        allow_mixed_endpoints=allow_mixed_endpoints
    )
    return validator.validate_file(file_path)


def quick_validate(file_path: Union[str, Path]) -> bool:
    """
    Quick validation check (returns True/False).

    Args:
        file_path: Path to the JSONL batch file

    Returns:
        True if valid, False otherwise

    Example:
        >>> if quick_validate("my_batch.jsonl"):
        ...     # Proceed with upload
        ...     pass
    """
    result = validate_batch_file(file_path)
    return result.is_valid
