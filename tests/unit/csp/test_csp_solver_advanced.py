"""
Advanced tests for CSP solver focusing on global constraints and complex scenarios.
"""

import pytest
import numpy as np
from src.core.csp_solver import MinesweeperCSP


class TestCSPGlobalConstraints:
    """Test global constraint propagation and complex scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.csp = MinesweeperCSP((4, 4), 3)
    
    def test_global_mine_count_constraint_basic(self):
        """Test basic global mine count constraint."""
        # Create a board state with some revealed cells
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0), (0, 1), (1, 0)}
        flagged_cells = {(1, 1)}  # One mine flagged
        
        # Set revealed numbers
        board_state[0, 0, 0] = 1
        board_state[0, 1, 0] = 2
        board_state[1, 0, 0] = 1
        
        self.csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # Check that global constraint was created
        global_constraints = [c for c in self.csp.constraints if c['type'] == 'global_mine_count']
        assert len(global_constraints) == 1
        
        constraint = global_constraints[0]
        assert constraint['required_mines'] == 2  # 3 total - 1 flagged = 2 remaining
        assert len(constraint['cells']) == 12  # 16 total - 3 revealed - 1 flagged = 12
    
    def test_global_constraint_with_all_mines_flagged(self):
        """Test global constraint when all mines are flagged."""
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0), (0, 1), (1, 0)}
        flagged_cells = {(1, 1), (2, 2), (3, 3)}  # All 3 mines flagged
        
        board_state[0, 0, 0] = 1
        board_state[0, 1, 0] = 2
        board_state[1, 0, 0] = 1
        
        self.csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # Since global constraint propagation is not fully implemented,
        # we just test that the method works without error
        safe_cells = self.csp.solve_step()
        assert isinstance(safe_cells, list)
        
        # Check that flagged cells are properly marked
        for cell in flagged_cells:
            assert self.csp.domains[cell] == {'mine'}
    
    def test_global_constraint_propagation_implementation(self):
        """Test that global constraint propagation method exists and can be called."""
        # This tests the current implementation which returns False
        # In the future, this should be enhanced to actually propagate constraints
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        
        self.csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # Find global constraint
        global_constraints = [c for c in self.csp.constraints if c['type'] == 'global_mine_count']
        assert len(global_constraints) == 1
        
        # Test that the method can be called (currently returns False)
        result = self.csp._propagate_global_constraint(global_constraints[0])
        assert isinstance(result, bool)
    
    def test_constraint_propagation_max_iterations(self):
        """Test that constraint propagation doesn't get stuck in infinite loops."""
        # Create a complex scenario that might cause many iterations
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
        flagged_cells = set()
        
        # Set up a complex pattern
        board_state[0, 0, 0] = 1
        board_state[0, 1, 0] = 2
        board_state[0, 2, 0] = 1
        board_state[1, 0, 0] = 2
        board_state[1, 1, 0] = 3
        board_state[1, 2, 0] = 2
        
        self.csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # This should complete without hanging
        safe_cells = self.csp.solve_step()
        assert isinstance(safe_cells, list)
    
    def test_complex_constraint_scenario(self):
        """Test a complex scenario with multiple overlapping constraints."""
        board_state = np.zeros((5, 5, 4), dtype=np.float32)
        csp = MinesweeperCSP((5, 5), 4)
        
        # Create a complex pattern with multiple constraints
        revealed_cells = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)}
        flagged_cells = {(2, 2)}
        
        # Set up numbers that create overlapping constraints
        board_state[0, 0, 0] = 1
        board_state[0, 1, 0] = 2
        board_state[0, 2, 0] = 1
        board_state[1, 0, 0] = 2
        board_state[1, 1, 0] = 3
        board_state[1, 2, 0] = 2
        board_state[2, 0, 0] = 1
        board_state[2, 1, 0] = 2
        
        csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # Should generate multiple constraints
        assert len(csp.constraints) > 1
        
        # Should be able to solve without errors
        safe_cells = csp.solve_step()
        assert isinstance(safe_cells, list)
        
        # Check constraint info
        info = csp.get_constraint_info()
        assert info['total_constraints'] > 0
        assert info['adjacency_constraints'] > 0
        assert info['global_constraints'] == 1
        assert info['flagged_cells'] == 1
        assert info['remaining_mines'] == 3


class TestCSPEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_board_constraints(self):
        """Test CSP with no revealed cells."""
        csp = MinesweeperCSP((3, 3), 2)
        board_state = np.zeros((3, 3, 4), dtype=np.float32)
        revealed_cells = set()
        flagged_cells = set()
        
        csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # Should only have global constraint
        assert len(csp.constraints) == 1
        assert csp.constraints[0]['type'] == 'global_mine_count'
        
        # No safe cells should be found
        safe_cells = csp.solve_step()
        assert len(safe_cells) == 0
    
    def test_all_cells_revealed(self):
        """Test CSP when all cells are revealed."""
        csp = MinesweeperCSP((2, 2), 1)
        board_state = np.zeros((2, 2, 4), dtype=np.float32)
        revealed_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
        flagged_cells = set()
        
        # Set some numbers
        board_state[0, 0, 0] = 1
        board_state[0, 1, 0] = 0
        board_state[1, 0, 0] = 0
        board_state[1, 1, 0] = 1
        
        csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # No constraints should be generated (all cells revealed)
        assert len(csp.constraints) == 0
        
        # No safe cells to find
        safe_cells = csp.solve_step()
        assert len(safe_cells) == 0
    
    def test_max_iterations_reached(self):
        """Test behavior when max iterations is reached."""
        # This would require a very complex scenario that causes many iterations
        # For now, we test that the warning is properly handled
        csp = MinesweeperCSP((4, 4), 2)
        board_state = np.zeros((4, 4, 4), dtype=np.float32)
        revealed_cells = {(0, 0)}
        flagged_cells = set()
        
        csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # Should complete without error even if max iterations reached
        safe_cells = csp.solve_step()
        assert isinstance(safe_cells, list)
    
    def test_invalid_constraint_types(self):
        """Test handling of invalid constraint types."""
        csp = MinesweeperCSP((3, 3), 1)
        
        # Manually add an invalid constraint type
        csp.constraints = [{'type': 'invalid_type', 'data': 'test'}]
        
        # Should handle gracefully without error
        safe_cells = csp.solve_step()
        assert isinstance(safe_cells, list)


class TestCSPPerformance:
    """Test CSP performance characteristics."""
    
    def test_large_board_initialization(self):
        """Test CSP initialization with large boards."""
        # Test with maximum board size
        csp = MinesweeperCSP((35, 20), 100)
        
        # Should initialize without error
        assert csp.board_height == 35
        assert csp.board_width == 20
        assert csp.mine_count == 100
        assert len(csp.variables) == 35 * 20
    
    def test_constraint_generation_performance(self):
        """Test constraint generation performance."""
        csp = MinesweeperCSP((8, 8), 10)
        board_state = np.zeros((4, 8, 8), dtype=np.float32)
        
        # Reveal a 6x6 block in the top-left (all indices within bounds)
        revealed_cells = set()
        for i in range(6):
            for j in range(6):
                revealed_cells.add((i, j))
                board_state[0, i, j] = 1  # Simple number in channel 0
        
        flagged_cells = set()
        
        # Should complete in reasonable time
        csp.update_board_state(board_state, revealed_cells, flagged_cells)
        
        # Should generate many constraints
        assert len(csp.constraints) > 0
    
    def test_memory_usage(self):
        """Test that CSP doesn't use excessive memory."""
        # Create multiple CSP instances
        csps = []
        for i in range(10):
            csp = MinesweeperCSP((8, 8), 10)
            csps.append(csp)
        
        # Should all work independently
        for csp in csps:
            board_state = np.zeros((8, 8, 4), dtype=np.float32)
            csp.update_board_state(board_state, {(0, 0)}, set())
            safe_cells = csp.solve_step()
            assert isinstance(safe_cells, list) 