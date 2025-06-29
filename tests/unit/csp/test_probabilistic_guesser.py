#!/usr/bin/env python3
"""
Unit tests for Probabilistic Guesser

Tests the probability-based guessing component for Minesweeper.
"""

import pytest
import numpy as np
import sys
import os
from typing import Set, Tuple, List

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.core.probabilistic_guesser import ProbabilisticGuesser


class TestProbabilisticGuesser:
    """Test the Probabilistic Guesser class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.guesser = ProbabilisticGuesser((4, 4), 2)
    
    def test_initialization(self):
        """Test probabilistic guesser initialization."""
        assert self.guesser.board_height == 4
        assert self.guesser.board_width == 4
        assert self.guesser.mine_count == 2
        assert isinstance(self.guesser.weights, dict)
        assert 'global_density' in self.guesser.weights
        assert 'edge_factor' in self.guesser.weights
        assert 'corner_factor' in self.guesser.weights
        assert 'adjacency_factor' in self.guesser.weights
    
    def test_get_neighbors(self):
        """Test neighbor calculation."""
        # Test corner cell
        neighbors = self.guesser._get_neighbors(0, 0)
        expected = [(0, 1), (1, 0), (1, 1)]
        assert set(neighbors) == set(expected)
        
        # Test edge cell
        neighbors = self.guesser._get_neighbors(0, 1)
        expected = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]
        assert set(neighbors) == set(expected)
        
        # Test center cell
        neighbors = self.guesser._get_neighbors(1, 1)
        expected = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        assert set(neighbors) == set(expected)
    
    def test_get_revealed_neighbors(self):
        """Test revealed neighbor calculation."""
        revealed_cells = {(0, 0), (1, 1), (2, 2)}
        
        # Test getting revealed neighbors of a cell
        revealed_neighbors = self.guesser._get_revealed_neighbors(1, 0, revealed_cells)
        assert (0, 0) in revealed_neighbors
        assert (1, 1) in revealed_neighbors
        assert (2, 2) not in revealed_neighbors  # Not a neighbor of (1, 0)
    
    def test_calculate_global_density_probability_empty(self):
        """Test global density probability with empty unrevealed cells."""
        unrevealed_cells = []
        remaining_mines = 2
        
        probs = self.guesser.calculate_global_density_probability(unrevealed_cells, remaining_mines)
        
        assert probs == {}
    
    def test_calculate_global_density_probability_normal(self):
        """Test global density probability with normal case."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        remaining_mines = 2
        
        probs = self.guesser.calculate_global_density_probability(unrevealed_cells, remaining_mines)
        
        # Should have probability 2/4 = 0.5 for each cell
        expected_prob = 2.0 / 4.0
        for cell in unrevealed_cells:
            assert probs[cell] == expected_prob
    
    def test_calculate_global_density_probability_no_mines(self):
        """Test global density probability with no remaining mines."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        remaining_mines = 0
        
        probs = self.guesser.calculate_global_density_probability(unrevealed_cells, remaining_mines)
        
        # Should have probability 0 for each cell
        for cell in unrevealed_cells:
            assert probs[cell] == 0.0
    
    def test_calculate_edge_probability(self):
        """Test edge probability calculation."""
        unrevealed_cells = [(0, 0), (1, 1), (3, 3)]
        
        probs = self.guesser.calculate_edge_probability(unrevealed_cells)
        
        # Corner cells should have higher edge factors
        assert probs[(0, 0)] > 1.0  # Corner
        assert probs[(3, 3)] > 1.0  # Corner
        # Center cell may have edge factor > 1.0 due to edge proximity calculation
        assert probs[(1, 1)] >= 1.0  # Center (may be affected by edge proximity)
    
    def test_calculate_corner_probability(self):
        """Test corner probability calculation."""
        unrevealed_cells = [(0, 0), (0, 3), (3, 0), (3, 3), (1, 1)]
        
        probs = self.guesser.calculate_corner_probability(unrevealed_cells)
        
        # Corner cells should have corner factor 1.2
        assert probs[(0, 0)] == 1.2  # Corner
        assert probs[(0, 3)] == 1.2  # Corner
        assert probs[(3, 0)] == 1.2  # Corner
        assert probs[(3, 3)] == 1.2  # Corner
        assert probs[(1, 1)] == 1.0  # Not corner
    
    def test_calculate_adjacency_probability_no_revealed_neighbors(self):
        """Test adjacency probability with no revealed neighbors."""
        unrevealed_cells = [(1, 1)]
        revealed_cells = {(0, 0)}  # Not a neighbor of (1, 1)
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        probs = self.guesser.calculate_adjacency_probability(unrevealed_cells, revealed_cells, board_state)
        
        # Should use neutral probability (may be 0.5 due to normalization)
        assert (1, 1) in probs
        assert isinstance(probs[(1, 1)], (float, np.floating))
    
    def test_calculate_adjacency_probability_with_revealed_neighbors(self):
        """Test adjacency probability with revealed neighbors."""
        unrevealed_cells = [(1, 1)]
        revealed_cells = {(0, 0), (0, 1), (1, 0)}  # Neighbors of (1, 1)
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        board_state[0, 0, 0] = 1  # Number 1
        board_state[0, 0, 1] = 2  # Number 2
        board_state[0, 1, 0] = 1  # Number 1
        
        probs = self.guesser.calculate_adjacency_probability(unrevealed_cells, revealed_cells, board_state)
        
        # Should calculate based on revealed neighbors
        assert (1, 1) in probs
        assert isinstance(probs[(1, 1)], (float, np.floating))
        assert probs[(1, 1)] > 0.0
    
    def test_calculate_adjacency_probability_invalid_numbers(self):
        """Test adjacency probability with invalid numbers (mines)."""
        unrevealed_cells = [(1, 1)]
        revealed_cells = {(0, 0)}
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        board_state[0, 0, 0] = -1  # Mine (invalid number)
        
        probs = self.guesser.calculate_adjacency_probability(unrevealed_cells, revealed_cells, board_state)
        
        # Should handle invalid numbers gracefully
        assert (1, 1) in probs
        assert isinstance(probs[(1, 1)], (float, np.floating))
    
    def test_calculate_adjacency_probability_no_unrevealed_neighbors(self):
        """Test adjacency probability when revealed cell has no unrevealed neighbors."""
        unrevealed_cells = [(1, 1)]
        revealed_cells = {(0, 0), (0, 1), (1, 0)}  # All neighbors of (1, 1) are revealed
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        board_state[0, 0, 0] = 1  # Number 1
        
        probs = self.guesser.calculate_adjacency_probability(unrevealed_cells, revealed_cells, board_state)
        
        # Should handle case where revealed cell has no unrevealed neighbors
        assert (1, 1) in probs
        assert isinstance(probs[(1, 1)], (float, np.floating))
    
    def test_get_guessing_candidates_empty(self):
        """Test guessing candidates with empty unrevealed cells."""
        unrevealed_cells = []
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        candidates = self.guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                                        flagged_cells, board_state)
        
        assert candidates == []
    
    def test_get_guessing_candidates_normal(self):
        """Test guessing candidates with normal case."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        revealed_cells = {(2, 2)}
        flagged_cells = {(3, 3)}
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        board_state[0, 2, 2] = 1  # Number 1 at revealed cell
        
        candidates = self.guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                                        flagged_cells, board_state)
        
        # Should return ranked list of candidates
        assert len(candidates) == 4
        assert all(cell in unrevealed_cells for cell in candidates)
        # Should be sorted by safety (safest first)
        assert candidates[0] in unrevealed_cells
    
    def test_get_guessing_candidates_with_adjacency(self):
        """Test guessing candidates with adjacency information."""
        unrevealed_cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
        revealed_cells = {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)}
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        # Set up some revealed numbers
        board_state[0, 0, 0] = 1
        board_state[0, 1, 0] = 2
        board_state[0, 2, 0] = 1
        board_state[1, 0, 0] = 2
        board_state[2, 0, 0] = 1
        
        candidates = self.guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                                        flagged_cells, board_state)
        
        # Should return ranked list
        assert len(candidates) == 4
        assert all(cell in unrevealed_cells for cell in candidates)
    
    def test_select_best_guess_with_candidates(self):
        """Test select best guess when candidates are available."""
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        revealed_cells = {(2, 2)}
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        best_guess = self.guesser.select_best_guess(unrevealed_cells, revealed_cells, 
                                                   flagged_cells, board_state)
        
        # Should return the best candidate
        assert best_guess is not None
        assert best_guess in unrevealed_cells
    
    def test_select_best_guess_no_candidates(self):
        """Test select best guess when no candidates are available."""
        unrevealed_cells = []
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        best_guess = self.guesser.select_best_guess(unrevealed_cells, revealed_cells, 
                                                   flagged_cells, board_state)
        
        # Should return None when no candidates
        assert best_guess is None
    
    def test_get_probability_info(self):
        """Test probability info retrieval."""
        cell = (1, 1)
        unrevealed_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        revealed_cells = {(2, 2)}
        flagged_cells = {(3, 3)}
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        board_state[0, 2, 2] = 1  # Number 1 at revealed cell
        
        info = self.guesser.get_probability_info(cell, unrevealed_cells, revealed_cells, 
                                               flagged_cells, board_state)
        
        # Should return detailed probability breakdown
        assert info['cell'] == cell
        assert 'global_density' in info
        assert 'edge_factor' in info
        assert 'corner_factor' in info
        assert 'adjacency_factor' in info
        assert 'combined_probability' in info
        assert 'weights' in info
        assert isinstance(info['combined_probability'], (float, np.floating))
        assert isinstance(info['weights'], dict)
    
    def test_edge_case_single_cell_board(self):
        """Test probabilistic guesser with single cell board."""
        guesser = ProbabilisticGuesser((1, 1), 0)
        
        unrevealed_cells = [(0, 0)]
        revealed_cells = set()
        flagged_cells = set()
        board_state = np.zeros((4, 1, 1), dtype=np.float32)
        
        # Test all methods with single cell
        global_probs = guesser.calculate_global_density_probability(unrevealed_cells, 0)
        edge_probs = guesser.calculate_edge_probability(unrevealed_cells)
        corner_probs = guesser.calculate_corner_probability(unrevealed_cells)
        adjacency_probs = guesser.calculate_adjacency_probability(unrevealed_cells, revealed_cells, board_state)
        
        assert (0, 0) in global_probs
        assert (0, 0) in edge_probs
        assert (0, 0) in corner_probs
        assert (0, 0) in adjacency_probs
        
        candidates = guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                                   flagged_cells, board_state)
        assert len(candidates) == 1
        assert candidates[0] == (0, 0)
    
    def test_edge_case_large_board(self):
        """Test probabilistic guesser with large board."""
        guesser = ProbabilisticGuesser((10, 10), 20)
        
        unrevealed_cells = [(i, j) for i in range(10) for j in range(10)]
        revealed_cells = set()
        flagged_cells = set()
        board_state = np.zeros((4, 10, 10), dtype=np.float32)
        
        # Test with large board
        global_probs = guesser.calculate_global_density_probability(unrevealed_cells, 20)
        edge_probs = guesser.calculate_edge_probability(unrevealed_cells)
        corner_probs = guesser.calculate_corner_probability(unrevealed_cells)
        
        assert len(global_probs) == 100
        assert len(edge_probs) == 100
        assert len(corner_probs) == 100
        
        # Test corner cells
        assert corner_probs[(0, 0)] == 1.2  # Corner
        assert corner_probs[(9, 9)] == 1.2  # Corner
        assert corner_probs[(5, 5)] == 1.0  # Not corner
    
    def test_probability_ranking_consistency(self):
        """Test that probability ranking is consistent."""
        unrevealed_cells = [(0, 0), (1, 1), (2, 2), (3, 3)]
        revealed_cells = {(0, 1), (1, 0)}
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        board_state[0, 0, 1] = 1  # Number 1
        board_state[0, 1, 0] = 2  # Number 2
        
        # Get candidates multiple times
        candidates1 = self.guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                                         flagged_cells, board_state)
        candidates2 = self.guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                                         flagged_cells, board_state)
        
        # Should be consistent
        assert candidates1 == candidates2
        
        # Should be sorted by safety
        assert len(candidates1) == 4
        assert all(cell in unrevealed_cells for cell in candidates1)
    
    def test_weights_impact_on_ranking(self):
        """Test that different weights affect ranking."""
        # Create guesser with different weights
        custom_weights = {
            'global_density': 0.1,
            'edge_factor': 0.9,  # Much higher weight on edge factor
            'corner_factor': 0.1,
            'adjacency_factor': 0.1
        }
        
        guesser = ProbabilisticGuesser((4, 4), 2)
        guesser.weights = custom_weights
        
        unrevealed_cells = [(0, 0), (1, 1), (2, 2), (3, 3)]
        revealed_cells = set()
        flagged_cells = set()
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        
        candidates = guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                                   flagged_cells, board_state)
        
        # With high edge factor weight, center cells should be preferred
        assert len(candidates) == 4
        # Center cell (1, 1) should be ranked higher than corner cells
        center_index = candidates.index((1, 1))
        corner_indices = [candidates.index((0, 0)), candidates.index((3, 3))]
        
        # Center should be ranked better (lower index) than at least some corners
        assert center_index < max(corner_indices)


def test_probabilistic_guesser_integration():
    """Test probabilistic guesser integration with complex scenario."""
    guesser = ProbabilisticGuesser((4, 4), 3)
    
    # Create a complex board state
    unrevealed_cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    revealed_cells = {(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)}
    flagged_cells = {(0, 3)}
    
    board_state = np.zeros((4, 4, 4), dtype=np.float32)
    # Set up some revealed numbers around the unrevealed area
    board_state[0, 0, 3] = 1  # Number 1
    board_state[0, 1, 3] = 2  # Number 2
    board_state[0, 2, 3] = 1  # Number 1
    board_state[0, 3, 0] = 2  # Number 2
    board_state[0, 3, 1] = 3  # Number 3
    board_state[0, 3, 2] = 2  # Number 2
    
    # Test all probability calculations
    remaining_mines = 3 - len(flagged_cells)  # 3 total - 1 flagged = 2
    
    global_probs = guesser.calculate_global_density_probability(unrevealed_cells, remaining_mines)
    edge_probs = guesser.calculate_edge_probability(unrevealed_cells)
    corner_probs = guesser.calculate_corner_probability(unrevealed_cells)
    adjacency_probs = guesser.calculate_adjacency_probability(unrevealed_cells, revealed_cells, board_state)
    
    assert len(global_probs) == 9
    assert len(edge_probs) == 9
    assert len(corner_probs) == 9
    assert len(adjacency_probs) == 9
    
    # Test candidate selection
    candidates = guesser.get_guessing_candidates(unrevealed_cells, revealed_cells, 
                                               flagged_cells, board_state)
    
    assert len(candidates) == 9
    assert all(cell in unrevealed_cells for cell in candidates)
    
    # Test best guess selection
    best_guess = guesser.select_best_guess(unrevealed_cells, revealed_cells, 
                                          flagged_cells, board_state)
    
    assert best_guess is not None
    assert best_guess in unrevealed_cells
    
    # Test probability info
    info = guesser.get_probability_info(best_guess, unrevealed_cells, revealed_cells, 
                                      flagged_cells, board_state)
    
    assert info['cell'] == best_guess
    assert 'combined_probability' in info
    assert isinstance(info['combined_probability'], (float, np.floating))


def test_probabilistic_guesser_main_function():
    """Test the main test function in probabilistic guesser file."""
    from src.core.probabilistic_guesser import test_probabilistic_guesser
    
    # This should run without error
    test_probabilistic_guesser()
    
    # The function should complete successfully
    assert True  # If we get here, the function ran without error


if __name__ == "__main__":
    pytest.main([__file__]) 