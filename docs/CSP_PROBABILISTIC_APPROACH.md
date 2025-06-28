# CSP + Probabilistic Hybrid Approach for Minesweeper

## Overview

This document outlines the implementation plan for a Constraint Satisfaction Problem (CSP) + Probabilistic guessing hybrid approach to solve Minesweeper. This approach combines deterministic logical reasoning with probability-based guessing when logical moves are exhausted.

## Current Baseline

- **RL Agent Performance**: 30% win rate on 4x4/2mines boards
- **Training Time**: 4.3 minutes for optimal performance
- **Stability**: Loss explosion issues resolved with learnable filtering

## Implementation Progress

### ✅ Phase 1: Setup & Infrastructure (Day 1) - **COMPLETED**

#### 1.1 Create CSP Branch ✅
```bash
git checkout -b feature/csp-probabilistic
git push -u origin feature/csp-probabilistic
```

#### 1.2 Create CSP Core Files ✅
- [x] `src/core/csp_solver.py` - Main CSP constraint solver
- [x] `src/core/probabilistic_guesser.py` - Probability-based guessing
- [x] `src/core/csp_agent.py` - Hybrid agent combining CSP + probability
- [x] `tests/unit/csp/test_csp_solver.py` - Unit tests for CSP
- [x] `tests/unit/csp/test_probabilistic_guesser.py` - Unit tests for probability

#### 1.3 Reuse Existing Infrastructure ✅
- [x] Adapt `minesweeper_env.py` for CSP integration (planned)
- [x] Reuse learnable filtering logic (planned)
- [x] Adapt testing framework for CSP evaluation (planned)
- [x] Reuse performance metrics and evaluation (planned)

**Status**: ✅ **COMPLETED** - All core files created and tested

### 🔄 Phase 2: CSP Solver Implementation (Day 2-3) - **IN PROGRESS**

#### 2.1 Basic CSP Structure ✅
- [x] Define variables (cells) and domains (safe/mine)
- [x] Implement adjacency constraints
- [x] Implement mine count constraints
- [x] Add constraint propagation logic

#### 2.2 CSP Solver Methods ✅
- [x] `solve_step()` - Find safe moves using constraint satisfaction
- [x] `get_constraints()` - Extract constraints from current board state
- [x] `propagate_constraints()` - Apply constraint propagation
- [x] `find_safe_cells()` - Identify cells that can be safely revealed

#### 2.3 Testing CSP Solver 🔄
- [x] Test on simple 4x4 boards with known solutions
- [ ] Test constraint propagation on edge cases
- [ ] Verify solver finds all logically safe moves
- [ ] Performance testing on various board sizes

**Status**: 🔄 **IN PROGRESS** - Basic CSP working, needs enhancement

### ⏳ Phase 3: Probabilistic Guessing (Day 4-5) - **PLANNED**

#### 3.1 Probability Models ⏳
- [ ] Mine density probability (global mine count)
- [ ] Edge probability (cells near board edges)
- [ ] Corner probability (corner cells)
- [ ] Adjacency probability (based on revealed neighbors)

#### 3.2 Guessing Logic ⏳
- [ ] `get_guessing_candidates()` - Return cells with lowest mine probability
- [ ] `calculate_cell_probability()` - Compute mine probability for each cell
- [ ] `rank_guessing_candidates()` - Sort cells by safety probability
- [ ] `select_best_guess()` - Choose optimal guessing candidate

#### 3.3 Integration with CSP ⏳
- [ ] Detect when CSP can't make progress
- [ ] Switch to probabilistic guessing
- [ ] Maintain CSP state during guessing
- [ ] Resume CSP solving after reveals

**Status**: ⏳ **PLANNED** - Algorithm designed, implementation pending

### ⏳ Phase 4: Hybrid Agent (Day 6-7) - **PLANNED**

#### 4.1 CSP Agent Implementation ⏳
- [ ] `CSPAgent` class combining CSP + probability
- [ ] `choose_action()` method with CSP → probability fallback
- [ ] State management and tracking
- [ ] Performance statistics collection

#### 4.2 Environment Integration ⏳
- [ ] Adapt environment to work with CSP agent
- [ ] Maintain compatibility with existing RL infrastructure
- [ ] Add CSP-specific evaluation metrics
- [ ] Update action selection logic

#### 4.3 Testing Framework ⏳
- [ ] Create comprehensive test script for CSP agent
- [ ] Performance comparison with current RL baseline
- [ ] Win rate evaluation on 4x4/2mines boards
- [ ] Statistical analysis of performance

**Status**: ⏳ **PLANNED** - Framework designed, implementation pending

### ⏳ Phase 5: Evaluation & Comparison (Day 8) - **PLANNED**

#### 5.1 Performance Testing ⏳
- [ ] Run CSP + Probabilistic on same test suite as RL
- [ ] Compare win rates: CSP vs RL (30% baseline)
- [ ] Analyze decision patterns and efficiency
- [ ] Document performance characteristics

#### 5.2 Analysis & Documentation ⏳
- [ ] Performance comparison report
- [ ] Decision pattern analysis
- [ ] Strengths/weaknesses of each approach
- [ ] Recommendations for next steps

**Status**: ⏳ **PLANNED** - Evaluation framework designed

### ⏳ Phase 6: Next Steps Planning (Day 9) - **PLANNED**

#### 6.1 Results Analysis ⏳
- [ ] If CSP + Probability > RL: Optimize and extend
- [ ] If CSP + Probability < RL: Plan CSP + RL hybrid
- [ ] If similar performance: Choose based on complexity/interpretability

#### 6.2 Future Implementation ⏳
- [ ] Plan CSP + RL hybrid if needed
- [ ] Design three-way comparison framework
- [ ] Outline curriculum learning integration
- [ ] Plan for larger board sizes

**Status**: ⏳ **PLANNED** - Analysis framework designed

## Probabilistic Algorithm Details

### Algorithm: Weighted Multi-Factor Probability Model

The probabilistic guesser uses a **weighted combination of four probability factors**:

#### 1. Global Mine Density Probability
```python
P(mine) = remaining_mines / unrevealed_cells
```
- **Weight**: 40% (0.4)
- **Logic**: Uniform distribution of remaining mines
- **Example**: 2 mines, 8 unrevealed cells → P(mine) = 0.25

#### 2. Edge Factor Probability
```python
edge_factor = 1.0 + (0.5 * (1.0 - dist_to_edge / max_dimension))
```
- **Weight**: 20% (0.2)
- **Logic**: Mines are more likely near edges
- **Example**: Corner cell gets factor ~1.5, center cell gets factor ~1.0

#### 3. Corner Factor Probability
```python
corner_factor = 1.2 if is_corner else 1.0
```
- **Weight**: 10% (0.1)
- **Logic**: Corner cells have different mine probabilities
- **Example**: Corner cells get 20% higher probability

#### 4. Adjacency Probability
```python
local_mine_density = revealed_number / unrevealed_neighbors
adjacency_factor = 0.5 + local_mine_density
```
- **Weight**: 30% (0.3)
- **Logic**: Use revealed numbers to estimate local mine density
- **Example**: Cell with "2" and 4 unrevealed neighbors → factor = 1.0

### Combined Probability Calculation
```python
combined_prob = (
    weight_global * global_prob +
    weight_edge * edge_prob +
    weight_corner * corner_prob +
    weight_adjacency * adjacency_prob
)
```

### Algorithm Advantages
- **Multi-factor**: Considers multiple sources of information
- **Weighted**: Prioritizes more reliable factors (global density, adjacency)
- **Adaptive**: Adjusts based on revealed information
- **Interpretable**: Each factor has clear reasoning

## Test Specifications

### Unit Tests

#### CSP Solver Tests (`test_csp_solver.py`)
- **Initialization Test**: Verify CSP creates correct variables and domains
- **Neighbor Calculation**: Test neighbor finding for corners, edges, center
- **Board State Update**: Test domain updates from revealed/flagged cells
- **Constraint Propagation**: Test simple constraint scenarios
- **Progress Detection**: Test `can_make_progress()` method
- **Constraint Info**: Test information retrieval methods

#### Probabilistic Guesser Tests (`test_probabilistic_guesser.py`)
- **Initialization Test**: Verify weights and board setup
- **Global Density**: Test uniform probability calculation
- **Edge Factor**: Test edge proximity probability
- **Corner Factor**: Test corner cell probability
- **Adjacency Factor**: Test local mine density estimation
- **Candidate Selection**: Test ranking and selection logic
- **Probability Info**: Test detailed probability breakdown

### Integration Tests (Planned)
- **CSP + Probability Integration**: Test fallback from CSP to probability
- **Environment Integration**: Test with actual Minesweeper environment
- **Performance Tests**: Test win rates and decision patterns
- **Comparison Tests**: Test against RL baseline

### Test Data Requirements
- **Simple Boards**: 4x4 with known solutions
- **Edge Cases**: Boards with no logical moves
- **Complex Scenarios**: Boards requiring multiple constraint steps
- **Performance Benchmarks**: Boards for timing analysis

## Success Criteria

### Minimum Viable Product
- [ ] CSP solver that finds all logical moves
- [ ] Probabilistic guessing that works when CSP can't progress
- [ ] Hybrid agent that achieves >25% win rate on 4x4/2mines
- [ ] Performance comparison with current RL baseline

### Stretch Goals
- [ ] >30% win rate (beat current RL baseline)
- [ ] Efficient CSP solving (<1 second per move)
- [ ] Interpretable decision making
- [ ] Extensible to larger board sizes

## Daily Checkpoints

- **Day 1**: ✅ CSP branch created, basic structure in place
- **Day 3**: 🔄 CSP solver working on simple boards
- **Day 5**: ⏳ Probabilistic guessing integrated
- **Day 7**: ⏳ Hybrid agent complete and tested
- **Day 8**: ⏳ Performance evaluation complete
- **Day 9**: ⏳ Results analysis and next steps planned

## Technical Details

### CSP Variables and Domains
- **Variables**: Each cell (i, j) on the board
- **Domains**: {safe, mine} for each cell
- **Constraints**: 
  - Adjacency constraints (revealed numbers)
  - Mine count constraints (total mines)
  - Revealed cell constraints (known safe/mine)

### Probability Models
- **Global Mine Density**: P(mine) = remaining_mines / unrevealed_cells
- **Edge Probability**: P(mine) = edge_mine_density * edge_factor
- **Corner Probability**: P(mine) = corner_mine_density * corner_factor
- **Adjacency Probability**: Based on revealed neighbor constraints

### Integration Strategy
1. **CSP First**: Always try to find logical moves
2. **Probability Fallback**: When CSP can't progress, use probability
3. **State Maintenance**: Keep CSP state updated during guessing
4. **Resume CSP**: After reveals, return to CSP solving

## Files Created

```
src/core/
├── csp_solver.py           # Main CSP constraint solver ✅
├── probabilistic_guesser.py # Probability-based guessing ✅
└── csp_agent.py           # Hybrid agent ✅

tests/unit/csp/
├── test_csp_solver.py     # CSP solver tests ✅
└── test_probabilistic_guesser.py # Probability tests ✅

scripts/
└── test_csp_performance.py # CSP performance testing ⏳
```

## Performance Metrics

- **Win Rate**: Primary success metric
- **Move Efficiency**: Average moves per game
- **CSP vs Probability Usage**: How often each method is used
- **Solving Time**: Time per move and per game
- **Decision Interpretability**: Can we explain why moves were chosen

## Comparison Framework

### Against Current RL Baseline
- **Win Rate**: Target >30%
- **Training Time**: CSP requires no training
- **Interpretability**: CSP decisions are explainable
- **Complexity**: CSP may be simpler to maintain

### Future CSP + RL Hybrid
- **CSP for logical moves**
- **RL for complex guessing patterns**
- **Best of both worlds approach**

---

**Status**: Phase 2 - CSP Solver Implementation (IN PROGRESS)
**Last Updated**: 2024-06-27
**Next Milestone**: Enhanced CSP constraint propagation and testing 