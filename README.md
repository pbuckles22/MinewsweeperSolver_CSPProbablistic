# Minesweeper CSP Probabilistic Solver

A clean, focused implementation of Constraint Satisfaction Problem (CSP) and probabilistic approaches for solving Minesweeper puzzles. This project provides systematic solving algorithms with comprehensive testing and 95% code coverage.

## Features

- **CSP Solving**: Constraint satisfaction problem solving for deterministic scenarios
- **Probabilistic Approach**: Advanced probability calculations for uncertain situations
- **Clean Environment**: Classic Minesweeper environment without unnecessary complexity
- **Comprehensive Testing**: 95% code coverage with 256 tests
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MinewsweeperSolver_CSPProbablistic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Run specific test categories
python -m pytest tests/unit/  # Unit tests only
python -m pytest tests/functional/  # Functional tests only
```

### Using the Solvers

```python
from src.core.minesweeper_env import MinesweeperEnv
from src.core.csp_agent import CSPAgent
from src.core.probabilistic_guesser import ProbabilisticGuesser

# Create environment
env = MinesweeperEnv(initial_board_size=(9, 9), initial_mines=10)
env.reset()

# Use CSP Agent
agent = CSPAgent(board_size=(9, 9), mine_count=10)
agent.update_state(env.state, set(), set())  # Update with current state
action = agent.choose_action()  # Get next move

# Use Probabilistic Guesser
guesser = ProbabilisticGuesser(board_size=(9, 9), mine_count=10)
# ... configure and use guesser
```

## Architecture

### Core Components

- **`minesweeper_env.py`**: Clean Minesweeper environment (95% coverage)
- **`csp_agent.py`**: CSP solver implementation (97% coverage)
- **`csp_solver.py`**: Core constraint satisfaction logic (90% coverage)
- **`probabilistic_guesser.py`**: Probabilistic approach (99% coverage)
- **`constants.py`**: Game constants and configuration (100% coverage)

### Test Structure

- **Unit Tests**: 256 comprehensive tests covering all components
- **Functional Tests**: End-to-end game flow testing
- **Performance Tests**: Algorithm efficiency validation
- **Edge Case Tests**: Robust error handling verification

## Performance

- **CSP Solver**: 4x4 boards (~0.001s), 9x9 boards (~0.01s), 16x16 boards (~0.1s)
- **Probabilistic Approach**: 85-95% success rate on uncertain scenarios
- **Test Performance**: Full test suite runs in ~0.5 seconds
- **Memory Usage**: Efficient data structures for large game states

## Development

### Code Standards

- Follow PEP 8 style guidelines
- Maintain 95%+ code coverage
- Write comprehensive tests for new features
- Use type hints where appropriate

### Testing Requirements

- **Minimum Coverage**: 85% for all new code
- **Target Coverage**: 95% for core components
- **Test Categories**: Unit, integration, functional, and performance tests
- **Test Maintenance**: Update tests when changing core functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass and coverage is maintained
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 