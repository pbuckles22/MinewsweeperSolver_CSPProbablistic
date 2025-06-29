#!/usr/bin/env python3
"""
Unit tests for CSP Agent

Tests the hybrid CSP + Probabilistic agent that combines constraint satisfaction
with probability-based guessing.
"""

import pytest
import numpy as np
from typing import Set, Tuple
from src.core.csp_agent import CSPAgent


class TestCSPAgent:
    """Test the CSP Agent class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.agent = CSPAgent((4, 4), 2)
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.board_size == (4, 4)
        assert self.agent.mine_count == 2
        assert self.agent.revealed_cells == set()
        assert self.agent.flagged_cells == set()
        assert self.agent.current_board_state is None
        assert self.agent.stats['total_moves'] == 0
        assert self.agent.stats['csp_moves'] == 0
        assert self.agent.stats['probability_moves'] == 0
        assert self.agent.stats['wins'] == 0
        assert self.agent.stats['losses'] == 0
        assert self.agent.stats['games_played'] == 0
    
    def test_reset(self):
        """Test agent reset functionality."""
        # Set some state
        self.agent.revealed_cells = {(0, 0), (1, 1)}
        self.agent.flagged_cells = {(2, 2)}
        self.agent.current_board_state = np.ones((4, 4, 4))
        self.agent.stats['csp_moves'] = 5
        self.agent.stats['probability_moves'] = 3
        
        # Reset
        self.agent.reset()
        
        # Check reset state
        assert self.agent.revealed_cells == set()
        assert self.agent.flagged_cells == set()
        assert self.agent.current_board_state is None
        # Stats should remain (reset only affects game state)
        assert self.agent.stats['csp_moves'] == 5
        assert self.agent.stats['probability_moves'] == 3
    
    def test_update_state(self):
        """Test state update functionality."""
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0), (1, 1)}
        flagged_cells = {(2, 2)}
        
        self.agent.update_state(board_state, revealed_cells, flagged_cells)
        
        assert self.agent.current_board_state is not None
        assert self.agent.revealed_cells == revealed_cells
        assert self.agent.flagged_cells == flagged_cells
        np.testing.assert_array_equal(self.agent.current_board_state, board_state)
    
    def test_get_unrevealed_cells(self):
        """Test unrevealed cells calculation."""
        # Set some revealed and flagged cells
        self.agent.revealed_cells = {(0, 0), (1, 1)}
        self.agent.flagged_cells = {(2, 2)}
        
        unrevealed = self.agent._get_unrevealed_cells()
        
        # Should have 13 unrevealed cells (16 total - 3 taken)
        assert len(unrevealed) == 13
        assert (0, 0) not in unrevealed
        assert (1, 1) not in unrevealed
        assert (2, 2) not in unrevealed
        assert (0, 1) in unrevealed
        assert (3, 3) in unrevealed
    
    def test_choose_action_no_board_state(self):
        """Test choose_action when no board state is available."""
        action = self.agent.choose_action()
        assert action is None
    
    def test_choose_action_no_unrevealed_cells(self):
        """Test choose_action when no unrevealed cells are available."""
        # Set board state but all cells revealed
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(i, j) for i in range(4) for j in range(4)}
        flagged_cells = set()
        
        self.agent.update_state(board_state, revealed_cells, flagged_cells)
        action = self.agent.choose_action()
        
        assert action is None
    
    def test_choose_action_csp_success(self):
        """Test choose_action when CSP finds safe moves."""
        # Create a board state where CSP can find safe moves
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        # Set up a simple scenario where CSP can work
        board_state[0, 0, 0] = 1  # Revealed number 1
        board_state[0, 1, 0] = 2  # Revealed number 2
        
        revealed_cells = {(0, 0), (0, 1)}
        flagged_cells = set()
        
        self.agent.update_state(board_state, revealed_cells, flagged_cells)
        action = self.agent.choose_action()
        
        # Should return a valid action
        assert action is not None
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert 0 <= action[0] < 4
        assert 0 <= action[1] < 4
        assert action not in revealed_cells
        assert action not in flagged_cells
    
    def test_choose_action_probability_fallback(self):
        """Test choose_action when CSP fails and probability is used."""
        # Create a board state where CSP can't find safe moves
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        # Set up a scenario where CSP can't make progress
        board_state[0, 0, 0] = 1  # Revealed number 1
        
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        
        self.agent.update_state(board_state, revealed_cells, flagged_cells)
        action = self.agent.choose_action()
        
        # Should return a valid action (probability-based)
        assert action is not None
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert 0 <= action[0] < 4
        assert 0 <= action[1] < 4
        assert action not in revealed_cells
        assert action not in flagged_cells
    
    def test_get_action_breakdown_no_moves(self):
        """Test action breakdown when no moves have been made."""
        breakdown = self.agent.get_action_breakdown()
        
        assert breakdown['total_moves'] == 0
        assert breakdown['csp_percentage'] == 0.0
        assert breakdown['probability_percentage'] == 0.0
        assert breakdown['csp_moves'] == 0
        assert breakdown['probability_moves'] == 0
    
    def test_get_action_breakdown_with_moves(self):
        """Test action breakdown when moves have been made."""
        # Simulate some moves
        self.agent.stats['csp_moves'] = 3
        self.agent.stats['probability_moves'] = 2
        
        breakdown = self.agent.get_action_breakdown()
        
        assert breakdown['total_moves'] == 5
        assert breakdown['csp_percentage'] == 60.0
        assert breakdown['probability_percentage'] == 40.0
        assert breakdown['csp_moves'] == 3
        assert breakdown['probability_moves'] == 2
    
    def test_get_csp_info(self):
        """Test CSP info retrieval."""
        csp_info = self.agent.get_csp_info()
        
        assert isinstance(csp_info, dict)
        # Should contain constraint information from CSP solver
        assert 'adjacency_constraints' in csp_info
        assert 'global_constraints' in csp_info
        assert 'flagged_cells' in csp_info
        assert 'remaining_mines' in csp_info
    
    def test_get_probability_info(self):
        """Test probability info retrieval."""
        # Set up board state
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        self.agent.update_state(board_state, revealed_cells, flagged_cells)
        
        # Get probability info for a cell
        cell = (1, 1)
        prob_info = self.agent.get_probability_info(cell)
        
        assert isinstance(prob_info, dict)
        # Should contain probability information
    
    def test_record_game_result_win(self):
        """Test recording a win."""
        initial_games = self.agent.stats['games_played']
        initial_wins = self.agent.stats['wins']
        
        self.agent.record_game_result(True)
        
        assert self.agent.stats['games_played'] == initial_games + 1
        assert self.agent.stats['wins'] == initial_wins + 1
        assert self.agent.stats['losses'] == 0
    
    def test_record_game_result_loss(self):
        """Test recording a loss."""
        initial_games = self.agent.stats['games_played']
        initial_losses = self.agent.stats['losses']
        
        self.agent.record_game_result(False)
        
        assert self.agent.stats['games_played'] == initial_games + 1
        assert self.agent.stats['losses'] == initial_losses + 1
        assert self.agent.stats['wins'] == 0
    
    def test_get_stats_no_games(self):
        """Test stats retrieval when no games have been played."""
        stats = self.agent.get_stats()
        
        assert 'win_rate' in stats
        assert stats['win_rate'] == 0.0
        assert 'total_moves' in stats
        assert 'csp_percentage' in stats
        assert 'probability_percentage' in stats
    
    def test_get_stats_with_games(self):
        """Test stats retrieval when games have been played."""
        # Record some game results
        self.agent.record_game_result(True)  # Win
        self.agent.record_game_result(False)  # Loss
        self.agent.record_game_result(True)   # Win
        
        # Add some moves
        self.agent.stats['csp_moves'] = 5
        self.agent.stats['probability_moves'] = 3
        
        stats = self.agent.get_stats()
        
        assert stats['games_played'] == 3
        assert stats['wins'] == 2
        assert stats['losses'] == 1
        assert stats['win_rate'] == (2/3) * 100
        assert stats['total_moves'] == 8
        assert stats['csp_percentage'] == (5/8) * 100
        assert stats['probability_percentage'] == (3/8) * 100
    
    def test_can_make_progress_true(self):
        """Test can_make_progress when progress is possible."""
        # Set up board state with unrevealed cells
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        self.agent.update_state(board_state, revealed_cells, flagged_cells)
        
        can_progress = self.agent.can_make_progress()
        assert can_progress is True
    
    def test_can_make_progress_false(self):
        """Test can_make_progress when no progress is possible."""
        # Set up board state with all cells revealed
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(i, j) for i in range(4) for j in range(4)}
        flagged_cells = set()
        self.agent.update_state(board_state, revealed_cells, flagged_cells)
        
        can_progress = self.agent.can_make_progress()
        assert can_progress is False
    
    def test_integration_full_game_simulation(self):
        """Test full integration with a simulated game."""
        agent = CSPAgent((3, 3), 1)
        
        # Simulate game progression
        board_state = np.zeros((4, 3, 3), dtype=np.float32)
        
        # First move - reveal corner
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        agent.update_state(board_state, revealed_cells, flagged_cells)
        
        action1 = agent.choose_action()
        assert action1 is not None
        
        # Simulate more moves
        revealed_cells.add(action1)
        agent.update_state(board_state, revealed_cells, flagged_cells)
        
        action2 = agent.choose_action()
        assert action2 is not None
        
        # Record game result
        agent.record_game_result(True)
        
        # Check final stats
        stats = agent.get_stats()
        assert stats['games_played'] == 1
        assert stats['wins'] == 1
        assert stats['win_rate'] == 100.0
        assert stats['total_moves'] >= 2  # At least 2 moves made


def test_csp_agent_edge_cases():
    """Test edge cases for CSP agent."""
    # Test with minimum board size
    agent = CSPAgent((1, 1), 0)
    assert agent.board_size == (1, 1)
    assert agent.mine_count == 0
    
    # Test with large board
    agent = CSPAgent((10, 10), 20)
    assert agent.board_size == (10, 10)
    assert agent.mine_count == 20
    
    # Test unrevealed cells on large board
    unrevealed = agent._get_unrevealed_cells()
    assert len(unrevealed) == 100  # 10x10 board
    assert (0, 0) in unrevealed
    assert (9, 9) in unrevealed


def test_csp_agent_statistics_edge_cases():
    """Test statistics edge cases."""
    agent = CSPAgent((2, 2), 1)
    
    # Test with only CSP moves
    agent.stats['csp_moves'] = 5
    agent.stats['probability_moves'] = 0
    
    breakdown = agent.get_action_breakdown()
    assert breakdown['csp_percentage'] == 100.0
    assert breakdown['probability_percentage'] == 0.0
    
    # Test with only probability moves
    agent.stats['csp_moves'] = 0
    agent.stats['probability_moves'] = 3
    
    breakdown = agent.get_action_breakdown()
    assert breakdown['csp_percentage'] == 0.0
    assert breakdown['probability_percentage'] == 100.0


def test_csp_agent_main_function():
    """Test the main test function in CSP agent file."""
    from src.core.csp_agent import test_csp_agent
    
    # This should run without error
    test_csp_agent()
    
    # The function should complete successfully
    assert True  # If we get here, the function ran without error 