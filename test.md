# Testing Documentation

## Overview

This project uses a comprehensive testing strategy with pytest to ensure code quality and reliability. The test suite covers unit tests, integration tests, functional tests, and script tests.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── unit/                          # Unit tests (537 tests)
│   ├── core/                      # Core environment unit tests
│   │   ├── test_core_action_space_unit.py
│   │   ├── test_core_actions_unit.py
│   │   ├── test_core_edge_cases_unit.py
│   │   ├── test_core_initialization_unit.py
│   │   ├── test_core_mechanics_unit.py
│   │   ├── test_core_state_unit.py
│   │   └── test_minesweeper_env.py
│   └── csp/                       # CSP solver unit tests
│       ├── test_csp_agent.py
│       ├── test_csp_solver.py
│       └── test_probabilistic_guesser.py
├── integration/                   # Integration tests (78 tests)
│   └── core/
├── functional/                    # Functional tests (112 tests)
│   ├── test_core_functional_requirements.py
│   ├── test_game_flow.py
│   └── test_performance.py
└── scripts/                       # Script tests (12 tests)
    ├── linux/
    ├── mac/
    └── windows/
```

## Test Categories

### Unit Tests (537 tests)
- **Core Environment Tests**: Test individual methods and properties of the Minesweeper environment
- **CSP Solver Tests**: Test constraint satisfaction problem solving logic
- **Probabilistic Guesser Tests**: Test probability-based decision making

### Integration Tests (78 tests)
- Test interactions between different components
- Verify that components work together correctly

### Functional Tests (112 tests)
- **Core Requirements**: Test that the system meets functional requirements
- **Game Flow**: Test complete game scenarios and workflows
- **Performance**: Test performance characteristics and benchmarks

### Script Tests (12 tests)
- Cross-platform script validation
- Platform-specific functionality testing

## Coverage Statistics

### Overall Coverage: 95%
- **minesweeper_env.py**: 95% coverage (34 missing lines)
- **csp_agent.py**: 97% coverage (3 missing lines)
- **csp_solver.py**: 90% coverage (14 missing lines)
- **probabilistic_guesser.py**: 99% coverage (1 missing line)
- **constants.py**: 100% coverage

### Missing Coverage Areas
The remaining uncovered lines are primarily in:
- Complex error handling paths
- Pygame rendering (when pygame is not available)
- Very specific edge cases
- Platform-specific code paths

## Quick Test Runner Scripts

### Automated Test Runner
The project includes convenient scripts for running the full test suite with coverage:

#### Python Script
```bash
# Run the Python test runner directly
python scripts/run_tests_with_coverage.py
```

#### Shell Script (Linux/Mac)
```bash
# Run the shell script wrapper
./scripts/run_tests.sh
```

#### Batch Script (Windows)
```cmd
# Run the Windows batch script
scripts\run_tests.bat
```

### Features of the Test Runner
- **Clean Output**: Shows only essential information
- **Coverage Summary**: Displays formatted coverage statistics
- **Color Coding**: Visual indicators for coverage levels
- **Error Handling**: Proper error reporting and exit codes
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Virtual Environment**: Automatically activates the correct environment

### Sample Output
```
🚀 Running Full Test Suite with Coverage...
============================================================

============================================================
📊 CODE COVERAGE SUMMARY
============================================================

📁 File Coverage:
--------------------------------------------------
🟢 src/__init__.py                       100% (0 missing)
🟢 src/core/__init__.py                  100% (0 missing)
🟢 src/core/constants.py                 100% (0 missing)
🟡 src/core/csp_agent.py                  97% (3 missing)
🟡 src/core/csp_solver.py                 90% (14 missing)
🟡 src/core/minesweeper_env.py            95% (34 missing)
🟡 src/core/probabilistic_guesser.py      99% (1 missing)

--------------------------------------------------
🟢 TOTAL COVERAGE:    95% (52 missing)
============================================================

✅ Test suite completed successfully!
🎉 Excellent coverage! (95%+)
```

## Running Tests

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Basic Test Commands

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Run specific test categories
python -m pytest tests/unit/                    # Unit tests only
python -m pytest tests/integration/             # Integration tests only
python -m pytest tests/functional/              # Functional tests only
python -m pytest tests/scripts/                 # Script tests only

# Run specific test files
python -m pytest tests/unit/core/test_minesweeper_env.py
python -m pytest tests/unit/csp/test_csp_agent.py

# Run with verbose output
python -m pytest -v

# Run with detailed failure information
python -m pytest -vv

# Run tests in parallel (if pytest-xdist is installed)
python -m pytest -n auto
```

### Platform-Specific Testing

#### Mac/Linux
```bash
# Run platform-specific scripts
python -m pytest tests/scripts/mac/
python -m pytest tests/scripts/linux/
```

#### Windows
```bash
# Run Windows-specific scripts
python -m pytest tests/scripts/windows/
```

### Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest --cov=src --cov-report=html

# Generate XML coverage report (for CI/CD)
python -m pytest --cov=src --cov-report=xml

# Generate coverage report with missing lines
python -m pytest --cov=src --cov-report=term-missing
```

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --timeout=30
markers =
    unit: Unit tests
    integration: Integration tests
    functional: Functional tests
    scripts: Script tests
    slow: Slow running tests
```

### Timeout Configuration
Tests have a 30-second timeout to prevent hanging tests. Long-running tests should be marked with `@pytest.mark.slow`.

## Test Fixtures

### Common Fixtures (conftest.py)
- **env**: Basic Minesweeper environment instance
- **csp_agent**: CSP agent instance
- **csp_solver**: CSP solver instance
- **probabilistic_guesser**: Probabilistic guesser instance

### Usage
```python
def test_example(env, csp_agent):
    # Use the fixtures in your tests
    assert env is not None
    assert csp_agent is not None
```

## Writing Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure
```python
import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv

class TestMinesweeperEnv:
    """Test the Minesweeper Environment class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.env = MinesweeperEnv()
    
    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.max_board_size == (35, 20)
        assert self.env.initial_board_size == (4, 4)
    
    def test_invalid_parameters(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Board size must be positive"):
            MinesweeperEnv(max_board_size=-5)
```

### Best Practices
1. **Arrange-Act-Assert**: Structure tests with clear sections
2. **Descriptive Names**: Use descriptive test and method names
3. **One Assertion**: Each test should test one specific behavior
4. **Edge Cases**: Include tests for edge cases and error conditions
5. **Documentation**: Add docstrings to test methods explaining what is being tested

## Continuous Integration

### GitHub Actions
The project includes GitHub Actions workflows that:
- Run tests on multiple Python versions
- Generate coverage reports
- Upload coverage to Codecov
- Run tests on different platforms (Linux, macOS, Windows)

### Pre-commit Hooks
Consider setting up pre-commit hooks to:
- Run tests before commits
- Check code formatting
- Validate imports
- Run linting tools

## Debugging Tests

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **Timeout Errors**: Mark slow tests with `@pytest.mark.slow`
3. **Platform Differences**: Use platform-specific test markers
4. **Random Failures**: Use fixed seeds for deterministic tests

### Debug Commands
```bash
# Run single test with debug output
python -m pytest tests/unit/core/test_minesweeper_env.py::TestMinesweeperEnv::test_initialization -v -s

# Run tests with print statements visible
python -m pytest -s

# Run tests with maximum verbosity
python -m pytest -vvv

# Run tests and stop on first failure
python -m pytest -x
```

## Performance Testing

### Benchmark Tests
Performance tests are located in `tests/functional/test_performance.py` and test:
- Environment initialization speed
- Game execution performance
- Memory usage patterns
- Algorithm efficiency

### Running Performance Tests
```bash
# Run performance tests only
python -m pytest tests/functional/test_performance.py -v

# Run with performance profiling
python -m pytest tests/functional/test_performance.py --profile
```

## Test Maintenance

### Adding New Tests
1. Create test file in appropriate directory
2. Follow naming conventions
3. Add comprehensive test coverage
4. Update this documentation if needed

### Updating Existing Tests
1. Ensure tests still pass after changes
2. Update test expectations if behavior changes
3. Add tests for new functionality
4. Remove obsolete tests

### Test Review Process
1. All new code should have corresponding tests
2. Aim for at least 85% code coverage
3. Review test quality and completeness
4. Ensure tests are maintainable and readable

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development) 