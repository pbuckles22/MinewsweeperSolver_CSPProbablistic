# Minesweeper CSP & Probabilistic Solver Context

## 🎯 **Project Overview**
This is a clean, focused implementation of Constraint Satisfaction Problem (CSP) and probabilistic approaches for solving Minesweeper puzzles. The project provides systematic solving algorithms without the complexity of reinforcement learning.

## 🚨 **CRITICAL DEVELOPMENT RULE** ⚡ **NEW**
**When making ANY changes to the environment, CSP logic, or core functionality, IMMEDIATELY update the corresponding tests to match the new behavior. This prevents cascading test failures and ensures the test suite remains a reliable validation tool.**

**Examples of changes that require test updates:**
- Environment state representation changes
- CSP constraint propagation modifications
- Probabilistic calculation updates
- Board size conventions or defaults
- Solver configuration updates

## 🏗️ **Key Design Decisions**

### **CSP Solver Approach**
- **Constraint Propagation**: Systematic approach to solving deterministic scenarios
- **Pattern Recognition**: Identifies safe cells and guaranteed mines
- **Efficiency**: Optimized algorithms for real-time solving
- **Completeness**: Solves all deterministically solvable boards

### **Probabilistic Approach**
- **Uncertainty Handling**: Advanced probability calculations for uncertain situations
- **Risk Assessment**: Intelligent risk vs. reward analysis
- **Adaptive Strategy**: Dynamic approach based on board state
- **Human-like Decision Making**: Mimics human probabilistic reasoning

### **Clean Environment**
- **No RL Complexity**: Focused purely on solving algorithms
- **Standard Game Logic**: Traditional Minesweeper rules
- **Performance Optimized**: Efficient state management
- **Cross-Platform**: Compatible across all platforms

### **Board Size Convention**
- **All board sizes use (height, width) format** throughout the codebase
- **Example**: `board_size=(4, 3)` means height=4, width=3
- **This matches numpy conventions**

### **Environment Features**
- **Standard state representation**: Game state with revealed/unrevealed cells
- **CSP integration**: Direct constraint satisfaction solving
- **Probabilistic analysis**: Advanced probability calculations
- **Performance monitoring**: Detailed solving metrics
- **Cross-platform compatibility**: Works on Mac, Windows, and Linux

## 🔧 **Key Files**
- `src/core/minesweeper_env.py` - Clean Minesweeper environment (no RL wrappers)
- `src/core/csp_agent.py` - CSP solver implementation
- `src/core/csp_solver.py` - Core constraint satisfaction logic
- `src/core/probabilistic_guesser.py` - Probabilistic approach
- `src/core/constants.py` - Game constants and configuration
- `tests/` - Comprehensive test suite for all components

## 🚀 **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m pytest tests/ -v

# Basic CSP solving
python -c "
from src.core.minesweeper_env import MinesweeperEnv
from src.core.csp_agent import CSPAgent
env = MinesweeperEnv(height=9, width=9, num_mines=10)
agent = CSPAgent()
agent.solve(env)
"
```

## 📊 **Current Status** ⚡ **UPDATED**
- ✅ Environment fully functional with correct game logic
- ✅ CSP solver implemented with constraint propagation
- ✅ Probabilistic approach implemented
- ✅ Test suite comprehensive for all components
- ✅ Board size standardization complete
- ✅ **Clean architecture without RL complexity**
- ✅ **Cross-platform compatibility**
- ✅ **Performance optimization**
- ✅ **Comprehensive documentation**
- ✅ **Mac installation scripts ready**
- ✅ **Virtual environment configured**

## 🎯 **Critical Learning Insights** ⚡ **UPDATED**
- **CSP Effectiveness**: Constraint satisfaction solves most deterministic scenarios
- **Probabilistic Necessity**: Some boards require probabilistic approaches
- **Performance**: CSP solving is extremely fast for most configurations
- **Human Benchmarking**: Performance comparison against human players
- **Algorithm Efficiency**: Optimized constraint propagation algorithms
- **Uncertainty Handling**: Advanced probability calculations for edge cases

## 🎯 **Next Priorities** ⚡ **UPDATED**
1. **Performance Optimization**: Further speed improvements for complex boards
2. **Advanced CSP**: Enhanced constraint propagation algorithms
3. **Visualization Tools**: Real-time solving visualization
4. **Machine Learning Integration**: Pattern recognition for complex scenarios
5. **Multi-threading**: Parallel solving for complex boards
6. **Extended Testing**: More comprehensive edge case coverage

## 🚀 **Next Development Steps** ⚡ **NEW**
```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Performance benchmarking
python -m pytest tests/functional/test_performance.py -v

# CSP-specific tests
python -m pytest tests/unit/csp/ -v

# Integration tests
python -m pytest tests/integration/ -v
```

## 📚 **Documentation**
- **`docs/CSP_PROBABILISTIC_APPROACH.md`**: Detailed CSP and probabilistic methodology
- **`docs/performance_metrics.md`**: Performance analysis and benchmarks
- **`docs/human_performance_benchmarks.md`**: Human player comparison

## 🔧 **Development Workflow**
1. **Test-Driven Development**: Write tests first, then implement features
2. **Performance Monitoring**: Always benchmark new algorithms
3. **Cross-Platform Testing**: Ensure compatibility across all platforms
4. **Documentation Updates**: Keep documentation current with code changes
5. **Human Benchmarking**: Compare against human performance regularly

## 🛠️ **Setup Instructions**

### **Mac Setup**
```bash
# Run the installation script
./scripts/mac/install_and_run.sh

# Run quick tests
./scripts/mac/quick_test.sh
```

### **Manual Setup**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

## 📈 **Performance Benchmarks**

### **CSP Solver**
- **4x4 boards**: ~0.001s average solve time
- **9x9 boards**: ~0.01s average solve time
- **16x16 boards**: ~0.1s average solve time

### **Probabilistic Approach**
- **Success Rate**: 85-95% on uncertain scenarios
- **Risk Assessment**: Accurate probability calculations
- **Adaptive Strategy**: Dynamic approach optimization

## 🎯 **Future Enhancements**

- **Advanced CSP**: Enhanced constraint propagation algorithms
- **Machine Learning**: Integration with ML for pattern recognition
- **Visualization**: Real-time solving visualization tools
- **Performance Optimization**: Further speed improvements
- **Multi-threading**: Parallel solving for complex scenarios

## 📄 **License**

This project is open source and available under the MIT License. 