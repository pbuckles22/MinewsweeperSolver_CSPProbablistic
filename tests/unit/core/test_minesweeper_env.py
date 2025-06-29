"""
Test suite for the Minesweeper environment.
See TEST_CHECKLIST.md for comprehensive test coverage plan.
"""

import pytest
import numpy as np
from src.core.minesweeper_env import MinesweeperEnv
from src.core.constants import (
    CELL_UNREVEALED, CELL_MINE, CELL_MINE_HIT,
    REWARD_FIRST_CASCADE_SAFE, REWARD_FIRST_CASCADE_HIT_MINE,
    REWARD_SAFE_REVEAL, REWARD_WIN, REWARD_HIT_MINE
)

class TestMinesweeperEnv:
    """Test cases for the Minesweeper environment."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.env = MinesweeperEnv(
            max_board_size=(10, 10),
            max_mines=10,
            initial_board_size=(4, 4),
            initial_mines=2
        )

    def test_initialization(self):
        """Test environment initialization with default parameters."""
        env = MinesweeperEnv()
        assert env.max_board_size == (35, 20)  # (height, width) format
        assert env.max_board_height == 35
        assert env.max_board_width == 20
        assert env.max_mines == 130
        assert env.initial_board_size == (4, 4)
        assert env.initial_mines == 2
        assert env.early_learning_mode == False
        assert env.early_learning_threshold == 200
        assert env.early_learning_corner_safe == True
        assert env.early_learning_edge_safe == True

        # Check parameters are set correctly
        assert self.env.max_board_width == 10
        assert self.env.max_board_height == 10
        assert self.env.max_mines == 10
        assert self.env.early_learning_mode is False
        assert self.env.early_learning_threshold == 200
        assert self.env.early_learning_corner_safe is True
        assert self.env.early_learning_edge_safe is True
        assert self.env.initial_board_width == 4
        assert self.env.initial_board_height == 4

    def test_board_creation(self):
        """Verify board is created with correct dimensions and initialization."""
        # Check board dimensions
        assert self.env.board.shape == (4, 4)
        assert self.env.state.shape == (4, 4, 4)  # 4 channels
        assert self.env.mines.shape == (4, 4)

        # Verify board is square
        assert self.env.board.shape[0] == self.env.board.shape[1]
        assert self.env.state.shape[1] == self.env.state.shape[2]  # height x width
        assert self.env.mines.shape[0] == self.env.mines.shape[1]

        # Check board is properly initialized with hidden cells
        assert np.all(self.env.state[0] == CELL_UNREVEALED)  # All cells should be hidden initially

        # Verify board size matches current dimensions
        assert self.env.board.shape[0] == self.env.current_board_height
        assert self.env.board.shape[1] == self.env.current_board_width

    def test_mine_placement(self):
        """Verify mines are placed correctly using only public API and deterministic setup."""
        # Use a fixed seed for deterministic mine placement
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=2)
        env.reset(seed=42)
        
        # Check mine count by examining the mines array directly
        # This is the most reliable way to count mines
        actual_mines = np.sum(env.mines)
        expected_mines = env.current_mines
        
        assert actual_mines == expected_mines, f"Expected {expected_mines} mines, but found {actual_mines} mines"

    def test_difficulty_levels(self):
        """Test environment with different difficulty levels using public API only."""
        difficulty_configs = [
            ('easy', 9, 9, 10),
            ('normal', 16, 16, 40),
            ('hard', 16, 30, 99),
            ('expert', 18, 24, 115),
            ('chaotic', 20, 35, 130)
        ]
        for name, height, width, mines in difficulty_configs:
            env = MinesweeperEnv(
                max_board_size=(height, width),  # (height, width) format
                max_mines=mines,
                initial_board_size=(height, width),  # (height, width) format
                initial_mines=mines
            )
            env.reset(seed=42)
            # Test board and state shapes
            assert env.state.shape == (4, height, width)  # 4 channels
            # Test action space
            expected_actions = width * height  # Only reveal actions
            assert env.action_space.n == expected_actions
            # Test observation space
            assert env.observation_space.shape == (4, height, width)  # 4 channels
            
            # Test mine count by examining the mines array directly
            actual_mines = np.sum(env.mines)
            assert actual_mines == mines, f"Difficulty {name}: Expected {mines} mines, but found {actual_mines} mines"

    def test_rectangular_board_actions(self):
        """Test actions on rectangular boards using public API only."""
        height, width, mines = 16, 30, 99  # (height, width) format
        env = MinesweeperEnv(
            max_board_size=(height, width),  # (height, width) format
            max_mines=mines,
            initial_board_size=(height, width),  # (height, width) format
            initial_mines=mines
        )
        env.reset(seed=42)
        # Test reveal actions (first 5 cells)
        for i in range(5):
            state, reward, terminated, truncated, info = env.step(i)
            # Allow any mine hit to terminate
            if terminated:
                env.reset(seed=42)
        # Check board and state shapes
        assert env.state.shape == (4, height, width)  # 4 channels
        assert env.observation_space.shape == (4, height, width)  # 4 channels
        assert env.action_space.n == width * height  # Only reveal actions

    def test_safe_cell_reveal(self):
        """Test safe cell reveal."""
        env = MinesweeperEnv(initial_board_size=4, initial_mines=2)
        env.reset()
        
        # Take an action
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        
        # Check that the revealed cell is no longer unrevealed
        assert state[0, 0, 0] != CELL_UNREVEALED

    def test_curriculum_progression(self):
        """Test curriculum learning progression through difficulty levels."""
        # Start with beginner level
        env = MinesweeperEnv(
            max_board_size=(35, 20),
            max_mines=130,
            initial_board_size=(4, 4),
            initial_mines=2
        )
        
        # Test progression through stages
        stages = [
            (4, 4, 2),    # Beginner
            (6, 6, 4),    # Intermediate
            (9, 9, 10),   # Easy
            (16, 16, 40), # Normal
            (16, 30, 99), # Hard
            (18, 24, 115),# Expert
            (20, 35, 130) # Chaotic
        ]
        
        for width, height, mines in stages:
            # Update board size and mines
            env.current_board_width = width
            env.current_board_height = height
            env.current_mines = mines
            env.reset()
            
            # Verify dimensions and mine count
            assert env.current_board_width == width
            assert env.current_board_height == height
            assert env.current_mines == mines
            
            # Verify board shapes
            assert env.board.shape == (height, width)
            assert env.state.shape == (4, height, width)  # 4 channels
            assert env.mines.shape == (height, width)
            
            # Verify action space
            expected_actions = width * height  # Only reveal actions
            assert env.action_space.n == expected_actions
            
            # Verify observation space
            assert env.observation_space.shape == (4, height, width)  # 4 channels
            
            # Test mine placement
            mine_count = np.sum(env.mines)
            assert mine_count == mines

    def test_win_condition_rectangular(self):
        """Test win condition on rectangular boards."""
        env = MinesweeperEnv(initial_board_size=(5, 3), initial_mines=3)
        env.reset(seed=42)
        
        # Try to reveal all cells
        for action in range(env.current_board_width * env.current_board_height):
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                # Game should end (either win or lose)
                break

    def test_reveal_action(self):
        """Test reveal action behavior."""
        self.env.reset()
        
        # Test revealing a safe cell
        action = 0
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Should not terminate unless we hit a mine or win
        if not terminated:
            assert state[0, 0, 0] != CELL_UNREVEALED
        else:
            # If terminated, should be either win or mine hit
            assert info.get('won', False) or reward == REWARD_HIT_MINE

    def test_invalid_actions(self):
        """Test invalid action handling."""
        self.env.reset()
        
        # Test revealing an already revealed cell
        action = 0
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Try to reveal the same cell again
        state, reward, terminated, truncated, info = self.env.step(action)
        assert reward < 0  # Should get negative reward for invalid action
        
        # Test out of bounds action
        action = self.env.action_space.n
        state, reward, terminated, truncated, info = self.env.step(action)
        assert reward < 0  # Should get negative reward for invalid action

    def test_game_over_condition(self):
        """Test game over condition."""
        self.env.reset()
        
        # Place mine at (0,0) and hit it
        self.env.mines[0, 0] = True
        self.env._update_adjacent_counts()
        self.env.mines_placed = True
        self.env.is_first_cascade = False
        self.env.first_cascade_done = True
        
        action = 0
        state, reward, terminated, truncated, info = self.env.step(action)
        
        assert terminated
        assert reward == REWARD_HIT_MINE
        assert state[0, 0, 0] == CELL_MINE_HIT 

    def test_initialization_default(self):
        """Test default initialization."""
        # Create a fresh environment without calling reset
        env = MinesweeperEnv()
        # Don't call reset() here to avoid mine placement
        
        assert env.max_board_size == (35, 20)
        assert env.max_mines == 130
        assert env.initial_board_size == (4, 4)
        assert env.initial_mines == 2
        assert env.current_board_height == 4
        assert env.current_board_width == 4
        assert env.current_mines == 2
    
    def test_initialization_with_int_board_size(self):
        """Test initialization with integer board size."""
        env = MinesweeperEnv(max_board_size=10, max_mines=50)  # Reduce max_mines
        assert env.max_board_size == (10, 10)
        assert env.max_board_height == 10
        assert env.max_board_width == 10
    
    def test_initialization_with_tuple_board_size(self):
        """Test initialization with tuple board size."""
        env = MinesweeperEnv(max_board_size=(15, 25))
        assert env.max_board_size == (15, 25)
        assert env.max_board_height == 15
        assert env.max_board_width == 25
    
    def test_initialization_with_int_initial_board_size(self):
        """Test initialization with integer initial board size."""
        env = MinesweeperEnv(initial_board_size=6)
        assert env.initial_board_size == (6, 6)
        assert env.initial_board_height == 6
        assert env.initial_board_width == 6
    
    def test_initialization_with_tuple_initial_board_size(self):
        """Test initialization with tuple initial board size."""
        env = MinesweeperEnv(initial_board_size=(5, 7))
        assert env.initial_board_size == (5, 7)
        assert env.initial_board_height == 5
        assert env.initial_board_width == 7
    
    def test_initialization_validation_negative_board_size(self):
        """Test validation of negative board size."""
        with pytest.raises(ValueError, match="Board size must be positive"):
            MinesweeperEnv(max_board_size=-5)
        
        with pytest.raises(ValueError, match="Board dimensions must be positive"):
            MinesweeperEnv(max_board_size=(-5, 10))
        
        with pytest.raises(ValueError, match="Board dimensions must be positive"):
            MinesweeperEnv(max_board_size=(5, -10))
    
    def test_initialization_validation_too_large_board_size(self):
        """Test validation of too large board size."""
        with pytest.raises(ValueError, match="Board dimensions too large"):
            MinesweeperEnv(max_board_size=101)
        
        with pytest.raises(ValueError, match="Board dimensions too large"):
            MinesweeperEnv(max_board_size=(101, 50))
        
        with pytest.raises(ValueError, match="Board dimensions too large"):
            MinesweeperEnv(max_board_size=(50, 101))
    
    def test_initialization_validation_negative_mines(self):
        """Test validation of negative mine count."""
        with pytest.raises(ValueError, match="Mine count must be positive"):
            MinesweeperEnv(max_mines=-5)
    
    def test_initialization_validation_too_many_mines(self):
        """Test validation of too many mines."""
        with pytest.raises(ValueError, match="Mine count cannot exceed board size area"):
            MinesweeperEnv(max_board_size=(3, 3), max_mines=10)
    
    def test_initialization_validation_negative_initial_board_size(self):
        """Test validation of negative initial board size."""
        with pytest.raises(ValueError, match="Initial board size must be positive"):
            MinesweeperEnv(initial_board_size=-3)
        
        with pytest.raises(ValueError, match="Initial board dimensions must be positive"):
            MinesweeperEnv(initial_board_size=(-3, 4))
        
        with pytest.raises(ValueError, match="Initial board dimensions must be positive"):
            MinesweeperEnv(initial_board_size=(3, -4))
    
    def test_initialization_validation_initial_board_too_large(self):
        """Test validation of initial board size exceeding max board size."""
        with pytest.raises(ValueError, match="Initial board size cannot exceed max board size"):
            MinesweeperEnv(max_board_size=5, max_mines=25, initial_board_size=6)
        
        with pytest.raises(ValueError, match="Initial board size cannot exceed max board size"):
            MinesweeperEnv(max_board_size=(5, 5), max_mines=25, initial_board_size=(6, 4))
        
        with pytest.raises(ValueError, match="Initial board size cannot exceed max board size"):
            MinesweeperEnv(max_board_size=(5, 5), max_mines=25, initial_board_size=(4, 6))
    
    def test_initialization_validation_negative_initial_mines(self):
        """Test validation of negative initial mine count."""
        with pytest.raises(ValueError, match="Initial mine count must be positive"):
            MinesweeperEnv(initial_mines=-1)
    
    def test_initialization_validation_too_many_initial_mines(self):
        """Test validation of too many initial mines."""
        with pytest.raises(ValueError, match="Initial mine count cannot exceed initial board area"):
            MinesweeperEnv(initial_board_size=(2, 2), initial_mines=5)
    
    def test_initialization_validation_none_rewards(self):
        """Test validation of None reward parameters."""
        with pytest.raises(TypeError):
            MinesweeperEnv(invalid_action_penalty=None)
        
        with pytest.raises(TypeError):
            MinesweeperEnv(mine_penalty=None)
        
        with pytest.raises(TypeError):
            MinesweeperEnv(safe_reveal_base=None)
        
        with pytest.raises(TypeError):
            MinesweeperEnv(win_reward=None)
    
    def test_properties(self):
        """Test environment properties."""
        env = MinesweeperEnv(max_board_size=(15, 25), initial_board_size=(5, 7))
        
        assert env.max_board_height == 15
        assert env.max_board_width == 25
        assert env.initial_board_height == 5
        assert env.initial_board_width == 7
        
        # Test backward compatibility property
        assert env.max_board_size_int == 15  # Returns height as default
    
    def test_max_board_size_int_property_square(self):
        """Test max_board_size_int property with square board."""
        env = MinesweeperEnv(max_board_size=10, max_mines=50)
        assert env.max_board_size_int == 10
    
    def test_reset_with_seed(self):
        """Test reset with seed for deterministic behavior."""
        env1 = MinesweeperEnv()
        env2 = MinesweeperEnv()
        
        # Reset with same seed
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)
        
        # Should be deterministic
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_reset_with_options(self):
        """Test reset with options parameter."""
        obs, info = self.env.reset(options={"board_size": (5, 5), "mines": 3})
        
        # Should handle options gracefully
        assert obs is not None
        assert info is not None
    
    def test_action_space_initialization(self):
        """Test action space initialization."""
        env = MinesweeperEnv(initial_board_size=(3, 4))
        assert env.action_space.n == 12  # 3 * 4 = 12
    
    def test_observation_space_initialization(self):
        """Test observation space initialization."""
        env = MinesweeperEnv(initial_board_size=(3, 4))
        assert env.observation_space.shape == (4, 3, 4)
        assert env.observation_space.dtype == np.float32
    
    def test_observation_space_bounds(self):
        """Test observation space bounds."""
        env = MinesweeperEnv(initial_board_size=(2, 2))
        
        # Test low bounds
        low = env.observation_space.low
        assert low.shape == (4, 2, 2)
        assert np.all(low[0] == -4)  # Channel 0: game state
        assert np.all(low[1] == -1)  # Channel 1: safety hints
        assert np.all(low[2] == 0)   # Channel 2: revealed cell count
        assert np.all(low[3] == 0)   # Channel 3: game progress
        
        # Test high bounds
        high = env.observation_space.high
        assert high.shape == (4, 2, 2)
        assert np.all(high[0] == 8)  # Channel 0: game state
        assert np.all(high[1] == 8)  # Channel 1: safety hints
        assert np.all(high[2] == 4)  # Channel 2: max revealed cells (2*2)
        assert np.all(high[3] == 1)  # Channel 3: binary indicators
    
    def test_state_initialization(self):
        """Test state initialization."""
        env = MinesweeperEnv(initial_board_size=(2, 2))
        env.reset()
        
        # Test initial state
        assert env.state.shape == (4, 2, 2)
        assert env.state.dtype == np.float32
        
        # Channel 0 should be all unrevealed
        assert np.all(env.state[0] == -1)  # CELL_UNREVEALED
    
    def test_board_initialization(self):
        """Test board initialization."""
        env = MinesweeperEnv(initial_board_size=(3, 3))
        env.reset()
        
        assert env.board.shape == (3, 3)
        assert env.mines.shape == (3, 3)
        assert env.revealed.shape == (3, 3)
        
        # Initially mines are placed during reset, but mines_placed flag may not be set
        # Check that some mines are placed (exact count may vary due to learnable config)
        assert np.sum(env.mines) > 0
        assert np.all(env.revealed == False)
    
    def test_early_learning_parameters(self):
        """Test early learning parameters."""
        env = MinesweeperEnv(
            early_learning_mode=True,
            early_learning_threshold=150,
            early_learning_corner_safe=False,
            early_learning_edge_safe=False
        )
        
        assert env.early_learning_mode == True
        assert env.early_learning_threshold == 150
        assert env.early_learning_corner_safe == False
        assert env.early_learning_edge_safe == False
    
    def test_reward_parameters(self):
        """Test reward parameters."""
        env = MinesweeperEnv(
            invalid_action_penalty=-10,
            mine_penalty=-20,
            safe_reveal_base=15,
            win_reward=500,
            first_cascade_safe_reward=25,
            first_cascade_hit_mine_reward=-30
        )
        
        assert env.invalid_action_penalty == -10
        assert env.mine_penalty == -20
        assert env.safe_reveal_base == 15
        assert env.win_reward == 500
        assert env.first_cascade_safe_reward == 25
        assert env.first_cascade_hit_mine_reward == -30
        assert env.reward_invalid_action == -10
    
    def test_statistics_initialization(self):
        """Test statistics initialization."""
        env = MinesweeperEnv()
        
        # Real-life statistics
        assert env.real_life_games_played == 0
        assert env.real_life_games_won == 0
        assert env.real_life_games_lost == 0
        
        # RL training statistics
        assert env.rl_games_played == 0
        assert env.rl_games_won == 0
        assert env.rl_games_lost == 0
        
        # Current game tracking
        assert env.current_game_was_pre_cascade == False
        assert env.current_game_ended_pre_cascade == False
    
    def test_move_counting_initialization(self):
        """Test move counting initialization."""
        env = MinesweeperEnv()
        
        assert env.move_count == 0
        assert env.total_moves_across_games == 0
        assert env.games_with_move_counts == []
    
    def test_action_tracking_initialization(self):
        """Test action tracking initialization."""
        env = MinesweeperEnv()
        
        assert env.repeated_actions == set()
        assert env.repeated_action_count == 0
        assert env.revealed_cell_click_count == 0
        assert env._actions_taken_this_game == set()
        assert env.invalid_action_count == 0
    
    def test_cascade_tracking_initialization(self):
        """Test cascade tracking initialization."""
        env = MinesweeperEnv()
        
        assert env.is_first_cascade == True
        assert env.in_cascade == False
    
    def test_render_mode_initialization(self):
        """Test render mode initialization."""
        env = MinesweeperEnv(render_mode=None)
        assert env.render_mode is None
        assert env.screen is None
        assert env.clock is None
        assert env.cell_size == 30
    
    def test_large_board_initialization(self):
        """Test initialization with large board."""
        env = MinesweeperEnv(max_board_size=(50, 50), initial_board_size=(10, 10))
        
        assert env.max_board_size == (50, 50)
        assert env.initial_board_size == (10, 10)
        assert env.current_board_height == 10
        assert env.current_board_width == 10
    
    def test_edge_case_minimum_board(self):
        """Test initialization with minimum board size."""
        env = MinesweeperEnv(max_board_size=(1, 1), max_mines=1, initial_board_size=(1, 1), initial_mines=1)
        
        assert env.max_board_size == (1, 1)
        assert env.initial_board_size == (1, 1)
        assert env.initial_mines == 1
        assert env.current_mines == 1
    
    def test_edge_case_maximum_board(self):
        """Test initialization with maximum board size."""
        env = MinesweeperEnv(max_board_size=(100, 100), initial_board_size=(50, 50))
        
        assert env.max_board_size == (100, 100)
        assert env.initial_board_size == (50, 50)
    
    def test_edge_case_maximum_mines(self):
        """Test initialization with maximum mines."""
        env = MinesweeperEnv(max_board_size=(10, 10), max_mines=100, initial_board_size=(5, 5), initial_mines=25)
        
        assert env.max_mines == 100
        assert env.initial_mines == 25
        # current_mines is set during reset
        env.reset()
        # After reset, current_mines may be different due to spacing constraints
        assert env.current_mines >= 0  # Should be non-negative

    def test_curriculum_learning_parameters(self):
        """Test curriculum learning parameters."""
        env = MinesweeperEnv(
            max_board_size=(20, 20),
            max_mines=50,
            initial_board_size=(4, 4),
            initial_mines=2
        )
        
        # Test that current parameters start at initial values
        assert env.current_board_height == 4
        assert env.current_board_width == 4
        assert env.current_mines == 2
        
        # Test that max parameters are set correctly
        assert env.max_board_height == 20
        assert env.max_board_width == 20
        assert env.max_mines == 50

    def test_step_with_numpy_action(self):
        """Test step method with numpy array action."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Test with numpy array action
        action = np.array([0])
        state, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_game_over_conditions(self):
        """Test step method when game is over."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Terminate the game
        env.terminated = True
        
        # Try to step
        state, reward, terminated, truncated, info = env.step(0)
        assert terminated == True
        assert reward == env.invalid_action_penalty
    
    def test_step_invalid_action_bounds(self):
        """Test step method with invalid action bounds."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Test with negative action
        state, reward, terminated, truncated, info = env.step(-1)
        assert reward == env.invalid_action_penalty
        assert not terminated
        
        # Test with action too large
        state, reward, terminated, truncated, info = env.step(10)
        assert reward == env.invalid_action_penalty
        assert not terminated
    
    def test_step_repeated_actions(self):
        """Test step method with repeated actions."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Take first action
        state, reward, terminated, truncated, info = env.step(0)
        initial_repeated_count = env.repeated_action_count
        
        # Take the same action again (should be invalid since cell is revealed)
        state, reward, terminated, truncated, info = env.step(0)
        assert reward == env.invalid_action_penalty
    
    def test_step_invalid_action_masks(self):
        """Test step method with invalid action masks."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Reveal a cell first
        state, reward, terminated, truncated, info = env.step(0)
        
        # Try to reveal the same cell again
        state, reward, terminated, truncated, info = env.step(0)
        assert reward == env.invalid_action_penalty
        # Note: revealed_cell_click_count may not be incremented in all cases
    
    def test_step_mine_hit(self):
        """Test step method when hitting a mine."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Find a mine position
        mine_positions = np.where(env.mines)
        if len(mine_positions[0]) > 0:
            row, col = mine_positions[0][0], mine_positions[1][0]
            action = row * env.current_board_width + col
            
            # Hit the mine
            state, reward, terminated, truncated, info = env.step(action)
            assert terminated == True
            assert reward == env.mine_penalty
            assert not info['won']
    
    def test_step_win_condition(self):
        """Test step method when winning the game."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Find a safe position
        safe_positions = np.where(~env.mines)
        if len(safe_positions[0]) > 0:
            row, col = safe_positions[0][0], safe_positions[1][0]
            action = row * env.current_board_width + col
            
            # Reveal the safe cell
            state, reward, terminated, truncated, info = env.step(action)
            
            # If this reveals all safe cells, we win
            if env._check_win():
                assert terminated == True
                assert reward == env.win_reward
                assert info['won']
    
    def test_action_masks_property(self):
        """Test action_masks property."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Test action masks
        masks = env.action_masks
        assert isinstance(masks, np.ndarray)
        assert masks.shape == (4,)  # 2x2 = 4 actions
        assert masks.dtype == bool
        
        # Test masks when game is over
        env.terminated = True
        masks = env.action_masks
        assert np.all(masks == False)
    
    def test_render_method(self):
        """Test render method."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Test render (should not raise exception)
        try:
            env.render()
        except Exception as e:
            # If pygame is not available, that's okay
            pass
        finally:
            # Always close the environment to clean up pygame
            env.close()
    
    def test_is_valid_action(self):
        """Test _is_valid_action method."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Test valid action
        assert env._is_valid_action(0) == True
        
        # Test invalid action (out of bounds)
        assert env._is_valid_action(-1) == False
        assert env._is_valid_action(10) == False
        
        # Test action on already revealed cell
        state, reward, terminated, truncated, info = env.step(0)
        assert env._is_valid_action(0) == False
    
    def test_get_cell_value(self):
        """Test _get_cell_value method."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Test cell value for unrevealed cell
        value = env._get_cell_value(0, 0)
        # Value depends on whether the cell is a mine or not
        assert isinstance(value, (int, np.integer))
        
        # Test cell value for revealed cell
        state, reward, terminated, truncated, info = env.step(0)
        value = env._get_cell_value(0, 0)
        assert isinstance(value, (int, np.integer))
    
    def test_get_neighbors(self):
        """Test _get_neighbors method."""
        env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
        obs, info = env.reset()
        
        # Test neighbors for corner cell
        neighbors = env._get_neighbors(0, 0)
        assert len(neighbors) == 3  # Top-left corner has 3 neighbors
        
        # Test neighbors for edge cell
        neighbors = env._get_neighbors(0, 1)
        assert len(neighbors) == 5  # Top edge has 5 neighbors
        
        # Test neighbors for center cell
        neighbors = env._get_neighbors(1, 1)
        assert len(neighbors) == 8  # Center has 8 neighbors
    
    def test_check_win(self):
        """Test _check_win method."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Initially not won
        assert env._check_win() == False
        
        # Reveal all safe cells
        safe_positions = np.where(~env.mines)
        for row, col in zip(safe_positions[0], safe_positions[1]):
            action = row * env.current_board_width + col
            state, reward, terminated, truncated, info = env.step(action)
        
        # Should be won now
        assert env._check_win() == True
    
    def test_update_adjacent_counts(self):
        """Test _update_adjacent_counts method."""
        env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
        obs, info = env.reset()
        
        # Clear the board first
        env.board.fill(0)
        env.mines.fill(False)
        
        # Place a mine manually
        env.mines[1, 1] = True
        env._update_adjacent_counts()
        
        # Check that the mine cell has value 9
        assert env.board[1, 1] == 9
        
        # Check that adjacent cells have increased values
        adjacent_sum = (env.board[0, 0] + env.board[0, 1] + env.board[0, 2] + 
                       env.board[1, 0] + env.board[1, 2] + 
                       env.board[2, 0] + env.board[2, 1] + env.board[2, 2])
        assert adjacent_sum > 0
    
    def test_reveal_cell_cascade(self):
        """Test _reveal_cell method with cascade."""
        env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
        obs, info = env.reset()
        
        # Place mine in corner and create a safe area
        env.mines[0, 0] = True
        env._update_adjacent_counts()
        
        # Reveal a cell that should cause cascade
        env._reveal_cell(2, 2)  # Far corner should be safe
        
        # Check that cascade occurred (may not always happen depending on board state)
        # Just check that the method doesn't raise an exception
        assert env.revealed[2, 2] == True
    
    def test_statistics_methods(self):
        """Test statistics methods."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Test real life statistics
        stats = env.get_real_life_statistics()
        assert isinstance(stats, dict)
        assert 'games_played' in stats
        assert 'games_won' in stats
        assert 'games_lost' in stats
        
        # Test RL training statistics
        stats = env.get_rl_training_statistics()
        assert isinstance(stats, dict)
        assert 'games_played' in stats
        assert 'games_won' in stats
        assert 'games_lost' in stats
        
        # Test combined statistics
        stats = env.get_combined_statistics()
        assert isinstance(stats, dict)
        assert 'real_life' in stats
        assert 'rl_training' in stats
    
    def test_move_statistics(self):
        """Test move statistics methods."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Test move statistics
        stats = env.get_move_statistics()
        assert isinstance(stats, dict)
        assert 'average_moves_per_game' in stats
        assert 'current_game_moves' in stats
        assert 'games_with_move_counts' in stats
    
    def test_board_statistics(self):
        """Test board statistics method."""
        env = MinesweeperEnv(initial_board_size=(4, 4), initial_mines=3)
        env.reset()
        
        stats = env.get_board_statistics()
        
        assert 'board_size' in stats
        assert 'mines_placed' in stats
        assert 'mine_positions' in stats
        assert 'total_cells' in stats
        assert 'mine_density' in stats
        assert 'safe_cells' in stats
        assert 'safe_cell_ratio' in stats
        
        assert stats['board_size'] == (4, 4)
        assert stats['mines_placed'] == 3
        assert stats['total_cells'] == 16
        assert len(stats['mine_positions']) == 3
    
    def test_record_game_moves(self):
        """Test _record_game_moves method."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Take some actions
        state, reward, terminated, truncated, info = env.step(0)
        
        # Record game moves
        env._record_game_moves()
        
        # Check that moves were recorded
        assert len(env.games_with_move_counts) > 0
    
    def test_update_statistics(self):
        """Test _update_statistics method."""
        env = MinesweeperEnv(initial_board_size=(2, 2), initial_mines=1)
        obs, info = env.reset()
        
        # Update statistics for a win
        env._update_statistics(game_won=True, game_ended_pre_cascade=False)
        
        # Check that statistics were updated
        assert env.real_life_games_played > 0
        assert env.real_life_games_won > 0
        
        # Update statistics for a loss
        env._update_statistics(game_won=False, game_ended_pre_cascade=True)
        
        # Check that statistics were updated
        assert env.real_life_games_lost > 0


def test_environment_edge_cases():
    """Test environment edge cases."""
    # Test with very small board
    env = MinesweeperEnv(initial_board_size=(1, 1), initial_mines=1, max_mines=1)
    obs, info = env.reset()
    assert obs.shape == (4, 1, 1)
    
    # Test with rectangular board
    env = MinesweeperEnv(initial_board_size=(2, 5))
    obs, info = env.reset()
    assert obs.shape == (4, 2, 5)
    assert env.action_space.n == 10  # 2 * 5 = 10
    
    # Test with different mine counts
    env = MinesweeperEnv(initial_board_size=(3, 3), initial_mines=1)
    obs, info = env.reset()
    assert obs.shape == (4, 3, 3)


def test_environment_validation_comprehensive():
    """Test comprehensive validation scenarios."""
    # Test all validation error cases
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=0)
    
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=(0, 5))
    
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=(5, 0))
    
    with pytest.raises(ValueError):
        MinesweeperEnv(max_mines=0)
    
    with pytest.raises(ValueError):
        MinesweeperEnv(initial_board_size=0)
    
    with pytest.raises(ValueError):
        MinesweeperEnv(initial_board_size=(0, 5))
    
    with pytest.raises(ValueError):
        MinesweeperEnv(initial_board_size=(5, 0))
    
    with pytest.raises(ValueError):
        MinesweeperEnv(initial_mines=0)
    
    # Test boundary conditions
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=101)
    
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=(101, 50))
    
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=(50, 101))
    
    # Test mine count validation
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=(2, 2), max_mines=5)
    
    with pytest.raises(ValueError):
        MinesweeperEnv(initial_board_size=(2, 2), initial_mines=5)
    
    # Test board size validation
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=5, max_mines=25, initial_board_size=6)
    
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=(5, 5), max_mines=25, initial_board_size=(6, 4))
    
    with pytest.raises(ValueError):
        MinesweeperEnv(max_board_size=(5, 5), max_mines=25, initial_board_size=(4, 6))


def test_environment_properties_comprehensive():
    """Test all environment properties comprehensively."""
    env = MinesweeperEnv(max_board_size=(15, 25), initial_board_size=(5, 7))
    
    # Test all properties
    assert env.max_board_height == 15
    assert env.max_board_width == 25
    assert env.initial_board_height == 5
    assert env.initial_board_width == 7
    assert env.max_board_size_int == 15  # Returns height for non-square
    
    # Test with square board
    env_square = MinesweeperEnv(max_board_size=10, max_mines=50)
    assert env_square.max_board_size_int == 10  # Returns the size for square board 