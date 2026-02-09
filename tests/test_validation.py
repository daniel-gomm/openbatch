"""Tests for batch file validation."""

import json
import pytest
from openbatch.validation import (
    validate_batch_file,
    quick_validate,
    ValidationResult,
)


@pytest.fixture
def temp_batch_file(tmp_path):
    """Provides a temporary file path for batch files."""
    return tmp_path / "test_batch.jsonl"


class TestValidationResult:
    def test_validation_result_str(self):
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            stats={"total_requests": 10, "file_size_mb": 0.5}
        )
        output = str(result)
        assert "FAILED" in output
        assert "Error 1" in output
        assert "Warning 1" in output
        assert "total_requests: 10" in output

    def test_validation_result_success(self):
        result = ValidationResult(is_valid=True, stats={"total_requests": 5})
        output = str(result)
        assert "PASSED" in output


class TestBatchFileValidator:
    def test_valid_batch_file(self, temp_batch_file):
        """Test validation of a valid batch file."""
        requests = [
            {
                "custom_id": "req_1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4", "input": "Hello"}
            },
            {
                "custom_id": "req_2",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4", "input": "World"}
            }
        ]

        with open(temp_batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.stats["total_requests"] == 2
        assert result.stats["unique_custom_ids"] == 2

    def test_file_not_found(self):
        """Test validation of non-existent file."""
        result = validate_batch_file("nonexistent.jsonl")
        assert not result.is_valid
        assert any("not found" in err.lower() for err in result.errors)

    def test_invalid_json(self, temp_batch_file):
        """Test validation of file with invalid JSON."""
        with open(temp_batch_file, "w") as f:
            f.write('{"custom_id": "req_1", "invalid json}\n')

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("invalid json" in err.lower() for err in result.errors)

    def test_duplicate_custom_ids(self, temp_batch_file):
        """Test detection of duplicate custom_ids."""
        requests = [
            {
                "custom_id": "req_1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4", "input": "Hello"}
            },
            {
                "custom_id": "req_1",  # Duplicate
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4", "input": "World"}
            }
        ]

        with open(temp_batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("duplicate" in err.lower() for err in result.errors)

    def test_missing_required_fields(self, temp_batch_file):
        """Test detection of missing required fields."""
        request = {
            "custom_id": "req_1",
            # Missing method, url, body
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("missing required fields" in err.lower() for err in result.errors)

    def test_invalid_method(self, temp_batch_file):
        """Test detection of invalid HTTP method."""
        request = {
            "custom_id": "req_1",
            "method": "GET",  # Should be POST
            "url": "/v1/responses",
            "body": {"model": "gpt-4", "input": "Hello"}
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("invalid method" in err.lower() for err in result.errors)

    def test_invalid_endpoint(self, temp_batch_file):
        """Test detection of invalid endpoint URL."""
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/invalid",  # Invalid endpoint
            "body": {"model": "gpt-4"}
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("invalid endpoint" in err.lower() for err in result.errors)

    def test_responses_api_missing_input(self, temp_batch_file):
        """Test Responses API validation - missing input/prompt."""
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": "gpt-4"}  # Missing input or prompt
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("input" in err.lower() or "prompt" in err.lower() for err in result.errors)

    def test_chat_completions_missing_messages(self, temp_batch_file):
        """Test Chat Completions API validation - missing messages."""
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4"}  # Missing messages
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("messages" in err.lower() for err in result.errors)

    def test_chat_completions_invalid_messages(self, temp_batch_file):
        """Test Chat Completions API validation - messages not an array."""
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4", "messages": "not an array"}
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("messages" in err.lower() and "array" in err.lower() for err in result.errors)

    def test_embeddings_missing_input(self, temp_batch_file):
        """Test Embeddings API validation - missing input."""
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {"model": "text-embedding-3-small"}  # Missing input
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("input" in err.lower() for err in result.errors)

    def test_mixed_endpoints_warning(self, temp_batch_file):
        """Test warning for mixed endpoint types."""
        requests = [
            {
                "custom_id": "req_1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4", "input": "Hello"}
            },
            {
                "custom_id": "req_2",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": "text-embedding-3-small", "input": "World"}
            }
        ]

        with open(temp_batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        result = validate_batch_file(temp_batch_file, allow_mixed_endpoints=False)
        assert result.is_valid  # Valid but with warning
        assert any("multiple endpoint" in warn.lower() for warn in result.warnings)

    def test_empty_lines_warning(self, temp_batch_file):
        """Test warning for empty lines."""
        with open(temp_batch_file, "w") as f:
            f.write('{"custom_id": "req_1", "method": "POST", "url": "/v1/responses", "body": {"model": "gpt-4", "input": "Hi"}}\n')
            f.write("\n")  # Empty line
            f.write('{"custom_id": "req_2", "method": "POST", "url": "/v1/responses", "body": {"model": "gpt-4", "input": "Bye"}}\n')

        result = validate_batch_file(temp_batch_file)
        assert any("empty line" in warn.lower() for warn in result.warnings)

    def test_wrong_file_extension_warning(self, tmp_path):
        """Test warning for wrong file extension."""
        json_file = tmp_path / "batch.json"  # Should be .jsonl
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": "gpt-4", "input": "Hello"}
        }

        with open(json_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(json_file)
        assert any(".jsonl" in warn.lower() for warn in result.warnings)

    def test_missing_model_in_body(self, temp_batch_file):
        """Test detection of missing model field in body."""
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {"input": "Hello"}  # Missing model
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("model" in err.lower() for err in result.errors)

    def test_invalid_custom_id_type(self, temp_batch_file):
        """Test detection of invalid custom_id type."""
        request = {
            "custom_id": 123,  # Should be string
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": "gpt-4", "input": "Hello"}
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("custom_id" in err.lower() for err in result.errors)

    def test_empty_custom_id(self, temp_batch_file):
        """Test detection of empty custom_id."""
        request = {
            "custom_id": "",  # Empty string
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": "gpt-4", "input": "Hello"}
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("custom_id" in err.lower() for err in result.errors)

    def test_body_not_object(self, temp_batch_file):
        """Test detection of non-object body."""
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/responses",
            "body": "not an object"
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert not result.is_valid
        assert any("body" in err.lower() and "object" in err.lower() for err in result.errors)

    def test_skip_custom_id_check(self, temp_batch_file):
        """Test disabling custom_id uniqueness check."""
        requests = [
            {
                "custom_id": "req_1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4", "input": "Hello"}
            },
            {
                "custom_id": "req_1",  # Duplicate
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4", "input": "World"}
            }
        ]

        with open(temp_batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        result = validate_batch_file(temp_batch_file, check_custom_id_uniqueness=False)
        # Should be valid when uniqueness check is disabled
        assert result.is_valid


class TestConvenienceFunctions:
    def test_quick_validate_true(self, temp_batch_file):
        """Test quick_validate with valid file."""
        request = {
            "custom_id": "req_1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": "gpt-4", "input": "Hello"}
        }

        with open(temp_batch_file, "w") as f:
            f.write(json.dumps(request) + "\n")

        assert quick_validate(temp_batch_file) is True

    def test_quick_validate_false(self, temp_batch_file):
        """Test quick_validate with invalid file."""
        with open(temp_batch_file, "w") as f:
            f.write("invalid json\n")

        assert quick_validate(temp_batch_file) is False


class TestComplexScenarios:
    def test_large_valid_file(self, temp_batch_file):
        """Test validation of file with many requests."""
        with open(temp_batch_file, "w") as f:
            for i in range(1000):
                request = {
                    "custom_id": f"req_{i}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {"model": "gpt-4", "input": f"Request {i}"}
                }
                f.write(json.dumps(request) + "\n")

        result = validate_batch_file(temp_batch_file)
        assert result.is_valid
        assert result.stats["total_requests"] == 1000
        assert result.stats["unique_custom_ids"] == 1000

    def test_all_three_endpoints(self, temp_batch_file):
        """Test file with all three valid endpoints."""
        requests = [
            {
                "custom_id": "req_1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4", "input": "Hello"}
            },
            {
                "custom_id": "req_2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
            },
            {
                "custom_id": "req_3",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": "text-embedding-3-small", "input": "Text"}
            }
        ]

        with open(temp_batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        result = validate_batch_file(temp_batch_file, allow_mixed_endpoints=True)
        assert result.is_valid
        assert len(result.stats["endpoints_used"]) == 3
