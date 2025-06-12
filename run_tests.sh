#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Run tests with coverage
pytest --cov=app tests/

# Check test coverage (optional)
# pytest-cov will exit with non-zero status if coverage is below threshold
# pytest --cov=app --cov-fail-under=80 tests/
