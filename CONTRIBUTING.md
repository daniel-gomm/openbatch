# Contributing to OpenBatch

Thank you for your interest in contributing to openbatch! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Branch Naming Convention](#branch-naming-convention)
- [Commit Message Convention](#commit-message-convention)
- [Running Tests](#running-tests)
- [Code Quality](#code-quality)
- [Continuous Integration](#continuous-integration)
- [Opening Issues](#opening-issues)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Documentation](#documentation)

## Code of Conduct

Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/openbatch.git
   cd openbatch
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/daniel-gomm/openbatch.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip
- git

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the package in editable mode with development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

   This ensures code quality checks run automatically before each commit.

### Verify Installation

```bash
# Run tests
pytest

# Check linting
ruff check src/ tests/

# Check formatting
ruff format --check src/ tests/

# Run type checking
mypy src/
```

## Development Workflow

1. **Sync your fork** with the upstream repository:
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

2. **Create a new branch** for your work (see [Branch Naming Convention](#branch-naming-convention)):
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit them (see [Commit Message Convention](#commit-message-convention))

4. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub

## Branch Naming Convention

We use the following prefixes for branch names:

- `feature/` - New features or enhancements
  - Example: `feature/add-retry-mechanism`
- `fix/` - Bug fixes
  - Example: `fix/validation-error-handling`
- `documentation/` - Documentation updates
  - Example: `documentation/improve-api-examples`

**Format**: `<prefix>/<short-description-with-hyphens>`

## Commit Message Convention

We use [Gitmoji](https://gitmoji.dev/) to prefix commit messages with relevant emojis that indicate the nature of the change.

### Common Gitmojis

| Emoji | Code | Description |
|-------|------|-------------|
| ✨ | `:sparkles:` | Introduce new features |
| 🐛 | `:bug:` | Fix a bug |
| 📝 | `:memo:` | Add or update documentation |
| ✅ | `:white_check_mark:` | Add, update, or pass tests |
| ♻️ | `:recycle:` | Refactor code |
| 🎨 | `:art:` | Improve structure/format of the code |
| ⚡️ | `:zap:` | Improve performance |
| 🔒️ | `:lock:` | Fix security issues |
| ⬆️ | `:arrow_up:` | Upgrade dependencies |
| ⬇️ | `:arrow_down:` | Downgrade dependencies |
| 🔧 | `:wrench:` | Add or update configuration files |
| 🚀 | `:rocket:` | Deploy stuff |
| 💚 | `:green_heart:` | Fix CI build |

See the full list at [gitmoji.dev](https://gitmoji.dev/).

### Commit Message Format

```
<gitmoji> <Short description>

[Optional detailed description]

[Optional footer with issue references]
```

### Examples

```bash
# Adding a new feature
git commit -m "✨ Add batch file validation module"

# Fixing a bug
git commit -m "🐛 Fix custom_id uniqueness check in validator"

# Updating documentation
git commit -m "📝 Add validation usage examples to README"

# Adding tests
git commit -m "✅ Add integration tests for validation module"

# Refactoring
git commit -m "♻️ Refactor BatchJobManager to use strategy pattern"
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=src/openbatch --cov-report=html
```

View the coverage report by opening `htmlcov/index.html` in your browser.

### Run Specific Tests

```bash
# Run a specific test file
pytest tests/test_validation.py

# Run a specific test class
pytest tests/test_validation.py::TestBatchFileValidator

# Run a specific test method
pytest tests/test_validation.py::TestBatchFileValidator::test_valid_batch_file

# Run tests matching a pattern
pytest -k "validation"
```

### Run Tests in Verbose Mode

```bash
pytest -v
```

## Code Quality

### Linting and Formatting

We use **Ruff** for both linting and formatting:

```bash
# Check for linting issues
ruff check src/ tests/

# Automatically fix linting issues
ruff check --fix src/ tests/

# Check code formatting
ruff format --check src/ tests/

# Format code
ruff format src/ tests/
```

### Type Checking

We use **mypy** for static type checking, configured for gradual adoption with reasonable strictness:

```bash
mypy src/
```

**Note**: Mypy is configured to allow `**kwargs` patterns and flexible union types that are common in this codebase. It focuses on catching real type errors while not being overly strict about edge cases.

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit:

- Trailing whitespace removal
- End-of-file fixer
- YAML syntax check
- Ruff linting and formatting
- Mypy type checking

To manually run all pre-commit hooks:

```bash
pre-commit run --all-files
```

## Continuous Integration

We use GitHub Actions for CI/CD. All pull requests must pass:

### Test Workflow (`.github/workflows/test.yml`)

- Runs on Python 3.11, 3.12, 3.13, 3.14
- Executes all tests with pytest
- Generates coverage report
- Uploads coverage to Codecov

### Lint Workflow (`.github/workflows/lint.yml`)

- Runs Ruff linting checks
- Runs Ruff formatting checks
- Runs mypy type checking (informational)

**All checks must pass before a PR can be merged.**

## Opening Issues

### Before Opening an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** to see if your question is already answered

### Issue Types

- **Bug Report**: Report a problem with the library
- **Feature Request**: Suggest a new feature or enhancement
- **Documentation**: Report issues with documentation
- **Question**: Ask questions about usage

### Bug Report Template

When reporting a bug, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**:
   ```python
   # Minimal code example
   ```
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - OS: (e.g., Ubuntu 22.04, Windows 11, macOS 13)
   - Python version: (e.g., 3.11.5)
   - OpenBatch version: (e.g., 0.0.4)
6. **Traceback** (if applicable)

### Feature Request Template

When requesting a feature:

1. **Problem**: Describe the problem your feature would solve
2. **Proposed Solution**: Describe your proposed solution
3. **Alternatives**: Alternative solutions you've considered
4. **Additional Context**: Any other context or examples

## Submitting Pull Requests

### Before Submitting

1. **Ensure all tests pass**: `pytest`
2. **Run code quality checks**: `ruff check src/ tests/`
3. **Format your code**: `ruff format src/ tests/`
4. **Update documentation** if you changed the API
5. **Add tests** for new features or bug fixes
6. **Update CHANGELOG.md** (if applicable)

### Pull Request Guidelines

1. **Title**: Use a clear, descriptive title
   - Good: "✨ Add validation for batch file size limits"
   - Bad: "Update validation.py"

2. **Description**: Include:
   - What changes you made and why
   - Related issue number (e.g., "Closes #123")
   - Screenshots/examples if applicable
   - Breaking changes (if any)

3. **Keep PRs focused**: One feature/fix per PR

4. **Write clear commit messages**: Follow the [Commit Message Convention](#commit-message-convention)

5. **Respond to feedback**: Be open to suggestions and discussions

### Pull Request Template

```markdown
## Description
<!-- Describe your changes -->

## Related Issue
<!-- Link to related issue: Closes #123 -->

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
<!-- Describe the tests you added/ran -->

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have followed the guidelines on commit messages
```

## Documentation

### Docstring Style

We use Google-style docstrings:

```python
def validate_batch_file(
    file_path: str | Path,
    strict: bool = True,
) -> ValidationResult:
    """
    Validate a batch file (convenience function).

    Args:
        file_path: Path to the JSONL batch file
        strict: Enable all checks (file size, request count)

    Returns:
        ValidationResult with errors, warnings, and statistics

    Raises:
        FileNotFoundError: If the file does not exist

    Example:
        >>> result = validate_batch_file("my_batch.jsonl")
        >>> if result.is_valid:
        ...     print("File is valid!")
    """
```

### Building Documentation

Documentation is built with MkDocs:

```bash
# Install documentation dependencies (if not already installed)
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

Visit `http://127.0.0.1:8000` to view the documentation.

### Adding Documentation Pages

1. Create a new markdown file in `docs/`
2. Add it to `mkdocs.yml` under `nav:`

## Questions?

If you have questions about contributing:

1. Check the [documentation](https://tiepnguyen2003.github.io/OpenAIBatchJobBuilder/)
2. Search [existing issues](https://github.com/TiepNguyen2003/OpenAIBatchJobBuilder/issues)
3. Open a new issue with the "Question" label

Thank you for contributing to OpenBatch! 🚀
