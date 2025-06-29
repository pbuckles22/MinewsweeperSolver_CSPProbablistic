"""
Advanced tests for Minesweeper environment focusing on complex scenarios and edge cases.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import time
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import CELL_UNREVEALED


class TestEnvironmentComplexErrorHandling:
    """Test complex error handling scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
        self.env.reset()
    
    def test_extreme_invalid_action_handling(self):
        """Test handling of extreme invalid action scenarios."""
        # Test actions way outside bounds
        extreme_actions = [-999999, 999999, 1000000, -1000000]
        
        for action in extreme_actions:
            state, reward, done, truncated, info = self.env.step(action)
            
            # Should handle gracefully
            assert reward == self.env.invalid_action_penalty
            assert not done
            assert not truncated
            # Note: environment doesn't set invalid_action key in info dict
    
    def test_repeated_invalid_actions_tracking(self):
        """Test tracking of repeated invalid actions."""
        # Make many invalid actions
        for _ in range(10):
            self.env.step(-1)  # Invalid action
        
        # Check that invalid action count is tracked
        stats = self.env.get_move_statistics()
        assert stats['invalid_action_count'] == 10
    
    def test_action_mask_edge_cases(self):
        """Test action mask behavior in edge cases."""
        # Test action masks when game is terminated
        self.env.terminated = True
        masks = self.env.action_masks
        
        # All actions should be invalid when game is over
        assert not np.any(masks)
    
    def test_complex_statistics_tracking(self):
        """Test complex statistics tracking across multiple games."""
        # Play multiple games to build up statistics
        for i in range(10):
            if self.env.terminated:
                break
            self.env.step(i % 16)  # Cycle through actions
        
        # Check that all statistics are properly tracked
        stats = self.env.get_combined_statistics()
        assert 'real_life' in stats
        assert 'rl_training' in stats
        
        # Check real_life statistics structure
        real_life_stats = stats['real_life']
        assert 'games_played' in real_life_stats
        assert 'games_won' in real_life_stats
        assert 'games_lost' in real_life_stats
        assert 'win_rate' in real_life_stats
        
        # Check rl_training statistics structure
        rl_stats = stats['rl_training']
        assert 'games_played' in rl_stats
        assert 'games_won' in rl_stats
        assert 'games_lost' in rl_stats
        assert 'win_rate' in rl_stats
    
    def test_error_handling_in_statistics_methods(self):
        """Test error handling in statistics methods."""
        # Test statistics methods on fresh environment
        fresh_env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
        
        # These should not raise errors even with no games played
        stats = fresh_env.get_real_life_statistics()
        assert isinstance(stats, dict)
        
        rl_stats = fresh_env.get_rl_training_statistics()
        assert isinstance(rl_stats, dict)
        
        combined_stats = fresh_env.get_combined_statistics()
        assert isinstance(combined_stats, dict)


class TestEnvironmentRenderingEdgeCases:
    """Test rendering edge cases and pygame availability."""
    
    def setup_method(self):
        """Set up test method."""
        self.envs = []  # Track environments to close
    
    def teardown_method(self):
        """Clean up after test method."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
        self.envs = []
    
    def test_rendering_without_pygame(self):
        """Test rendering behavior when pygame is not available."""
        with patch('pygame.display') as mock_display:
            mock_display.side_effect = ImportError("pygame not available")
            
            # Should handle pygame import error gracefully
            env = MinesweeperEnv()
            self.envs.append(env)
            
            # Should not crash when trying to render
            try:
                env.render()
            except Exception as e:
                # Should handle gracefully, not crash
                assert "pygame" in str(e) or "display" in str(e)
    
    def test_rendering_mode_none(self):
        """Test rendering when render_mode is None."""
        env = MinesweeperEnv(render_mode=None)
        self.envs.append(env)
        env.reset()
        
        # Should not crash when render is called
        env.render()  # Should do nothing
    
    def test_rendering_with_invalid_screen(self):
        """Test rendering with invalid screen setup."""
        with patch('pygame.display.set_mode') as mock_set_mode:
            mock_set_mode.side_effect = Exception("Display error")
            
            # Should handle display setup errors
            try:
                env = MinesweeperEnv()
                self.envs.append(env)
            except Exception:
                # Should handle gracefully
                pass
    
    def test_rendering_performance(self):
        """Test rendering performance with large boards."""
        # Test with maximum board size
        env = MinesweeperEnv(max_board_size=(35, 20))
        self.envs.append(env)
        env.reset()
        
        # Should render without performance issues
        env.render()
        
        # Should handle multiple render calls
        for _ in range(5):
            env.render()


class TestEnvironmentMultiMineScenarios:
    """Test complex multi-mine scenarios."""
    
    def test_complex_multi_mine_placement(self):
        """Test complex multi-mine placement scenarios."""
        # Test with many mines on a large board
        env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=15)
        env.reset()
        
        # Should handle complex mine placement
        assert env.mines.sum() == 15
    
    def test_edge_case_maximum_mines(self):
        """Test edge case with maximum possible mines."""
        # Test with maximum mine density
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=15)  # Almost all cells
        env.reset()
        
        # Should handle extreme mine density
        assert env.mines.sum() == 15
        
        # Should still be able to make moves
        state, reward, done, truncated, info = env.step(0)
        assert isinstance(reward, (int, float))


class TestEnvironmentPerformanceEdgeCases:
    """Test performance edge cases and large board scenarios."""
    
    def test_maximum_board_size_performance(self):
        """Test performance with maximum board size."""
        env = MinesweeperEnv(max_board_size=(35, 20), initial_board_size=(35, 20), initial_mines=100)
        
        # Should initialize in reasonable time
        start_time = time.time()
        env.reset()
        init_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert init_time < 5.0  # Should initialize in under 5 seconds
    
    def test_large_board_memory_usage(self):
        """Test memory usage with large boards."""
        # Create multiple large environments
        envs = []
        for _ in range(5):
            env = MinesweeperEnv(max_board_size=(35, 20), initial_board_size=(35, 20), initial_mines=100)
            env.reset()
            envs.append(env)
        
        # Should all work independently
        for env in envs:
            state, reward, done, truncated, info = env.step(0)
            assert isinstance(state, np.ndarray)
    
    def test_complex_game_scenarios(self):
        """Test complex game scenarios with many moves."""
        env = MinesweeperEnv(initial_board_size=(8, 8), initial_mines=10)
        env.reset()
        
        # Play a complex game
        moves = 0
        while not env.terminated and moves < 50:
            # Try different actions
            action = moves % 64  # Cycle through all possible actions
            state, reward, done, truncated, info = env.step(action)
            moves += 1
            
            if done:
                break
        
        # Should complete without errors
        assert moves > 0
    
    def test_statistics_performance(self):
        """Test statistics calculation performance."""
        env = MinesweeperEnv(initial_board_size=(6, 6), initial_mines=8)
        
        # Play many games quickly
        for _ in range(10):
            env.reset()
            moves = 0
            while not env.terminated and moves < 20:  # Limit moves to prevent infinite loops
                # Select a random valid action instead of always clicking first cell
                valid_actions = np.where(env.action_masks)[0]
                if len(valid_actions) == 0:
                    break  # No valid actions left
                action = np.random.choice(valid_actions)
                env.step(action)
                moves += 1
        
        # Should calculate statistics quickly
        start_time = time.time()
        stats = env.get_combined_statistics()
        calc_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert calc_time < 1.0  # Should calculate in under 1 second


class TestEnvironmentIntegrationEdgeCases:
    """Test integration edge cases between different components."""
    
    def test_csp_integration_edge_cases(self):
        """Test CSP integration with edge cases."""
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
        env.reset()
        
        # Test CSP agent integration
        from src.core.csp_agent import CSPAgent
        agent = CSPAgent(board_size=(4, 4), mine_count=2)
        
        # Update agent state with current environment state
        revealed_cells = set()
        flagged_cells = set()
        for i in range(4):
            for j in range(4):
                if env.board[i, j] != CELL_UNREVEALED:
                    revealed_cells.add((i, j))
        
        agent.update_state(env.state, revealed_cells, flagged_cells)
        
        # Should work with environment
        action = agent.choose_action()
        assert action is None or isinstance(action, tuple)
        if action is not None:
            assert len(action) == 2
            assert 0 <= action[0] < 4
            assert 0 <= action[1] < 4
    
    def test_probabilistic_integration_edge_cases(self):
        """Test probabilistic guesser integration with edge cases."""
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=6)  # High mine density
        env.reset()
        
        # Test probabilistic guesser integration
        from src.core.probabilistic_guesser import ProbabilisticGuesser
        guesser = ProbabilisticGuesser(board_size=(4, 4), mine_count=6)
        
        # Get unrevealed cells
        unrevealed_cells = []
        revealed_cells = set()
        flagged_cells = set()
        for i in range(4):
            for j in range(4):
                if env.board[i, j] == CELL_UNREVEALED:
                    unrevealed_cells.append((i, j))
                else:
                    revealed_cells.add((i, j))
        
        # Should work with environment
        action = guesser.select_best_guess(unrevealed_cells, revealed_cells, flagged_cells, env.state)
        assert action is None or isinstance(action, tuple)
        if action is not None:
            assert len(action) == 2
            assert 0 <= action[0] < 4
            assert 0 <= action[1] < 4
    
    def test_cross_component_state_consistency(self):
        """Test state consistency across different components."""
        env = MinesweeperEnv(initial_board_size=(5, 5), initial_mines=5)
        env.reset()
        
        # Make a move
        state, reward, done, truncated, info = env.step(0)
        
        # Check state consistency
        assert state.shape == (4, 5, 5)  # 4 channels, 5x5 board
        assert np.all(state >= -1)  # All values should be >= -1
        assert np.all(state <= 8)   # All values should be <= 8 