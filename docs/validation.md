# Batch File Validation

OpenBatch includes comprehensive validation to catch errors before uploading batch files to OpenAI, saving time and preventing failed uploads.

## Overview

The validation module checks your batch files for:

- ✓ Valid JSONL format
- ✓ Unique `custom_id` values
- ✓ Required fields present
- ✓ Valid HTTP methods (POST)
- ✓ Valid endpoint URLs
- ✓ Correct request body structure for each API type
- ✓ File size limits (200 MB)
- ✓ Request count limits (50,000)
- ⚠ Mixed endpoint types warning

## Simple Validation

```python
from openbatch import validate_batch_file

result = validate_batch_file("my_batch.jsonl")

if result.is_valid:
    print(f"✓ Batch file is valid!")
    print(f"Total requests: {result.stats['total_requests']}")
    # Proceed with upload to OpenAI
else:
    print(f"✗ Validation failed:")
    print(result)  # Shows detailed errors and warnings
```

### Quick Boolean Check

For a fast True/False check:

```python
from openbatch.validation import quick_validate

if quick_validate("my_batch.jsonl"):
    # File is valid, proceed
    upload_to_openai("my_batch.jsonl")
```

## Validation Result

The `ValidationResult` object provides detailed information:

```python
result = validate_batch_file("batch.jsonl")

# Check validity
if result.is_valid:
    # Access statistics
    print(f"Requests: {result.stats['total_requests']}")
    print(f"File size: {result.stats['file_size_mb']} MB")
    print(f"Endpoints: {result.stats['endpoints_used']}")
    print(f"Unique IDs: {result.stats['unique_custom_ids']}")
else:
    # Handle errors
    for error in result.errors:
        print(f"ERROR: {error}")

    # Review warnings
    for warning in result.warnings:
        print(f"WARNING: {warning}")
```

### Human-Readable Output

The `ValidationResult` has a nice string representation:

```python
result = validate_batch_file("batch.jsonl")
print(result)
```

Output:
```
Validation: PASSED

Statistics:
  total_requests: 100
  unique_custom_ids: 100
  endpoints_used: ['/v1/responses']
  file_size_mb: 0.05

Warnings (1):
  • File extension is '.json', expected '.jsonl'
```

## Advanced Usage

### Configurable Validation

Use `BatchFileValidator` for custom validation options:

```python
from openbatch.validation import BatchFileValidator

validator = BatchFileValidator(
    check_custom_id_uniqueness=True,   # Check for duplicate IDs
    check_file_size=True,               # Check 200 MB limit
    check_request_count=True,           # Check 50K request limit
    allow_mixed_endpoints=False         # Warn about mixed endpoints
)

result = validator.validate_file("batch.jsonl")
```

### Disable Specific Checks

```python
from openbatch import validate_batch_file

# Skip custom_id uniqueness check (if you want duplicates)
result = validate_batch_file(
    "batch.jsonl",
    check_custom_id_uniqueness=False
)

# Allow mixed endpoints without warning
result = validate_batch_file(
    "batch.jsonl",
    allow_mixed_endpoints=True
)

# Disable strict checks (file size, request count)
result = validate_batch_file(
    "batch.jsonl",
    strict=False
)
```
