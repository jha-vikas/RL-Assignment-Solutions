"""
GridWorld Value Iteration Implementation
=========================================
4x4 GridWorld where agent navigates from top-left (0) to bottom-right (15)
using Value Iteration with the Bellman equation.
"""

import numpy as np

# ============================================================================
# 1. INITIALIZATION
# ============================================================================

# Grid configuration
N = 4  # Grid size (NxN)
NUM_STATES = N * N
TERMINAL_STATE = NUM_STATES - 1  # Bottom-right corner (state 15)

# Rewards
MOVE_REWARD = -1  # Reward for each move
TERMINAL_REWARD = 0  # Reward at terminal state

# Value iteration parameters
GAMMA = 1.0  # Discount factor (no discounting)
THETA = 1e-4  # Convergence threshold

# Initialize value function to zeros
V = np.zeros(NUM_STATES)

# ============================================================================
# 2. DEFINE ACTIONS
# ============================================================================

# Actions: [row_delta, col_delta]
ACTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def state_to_position(state):
    """Convert state number to (row, col) position."""
    return state // N, state % N

def position_to_state(row, col):
    """Convert (row, col) position to state number."""
    return row * N + col

def get_next_state(state, action):
    """
    Get next state given current state and action.
    If action leads outside grid, stay in current state.
    """
    row, col = state_to_position(state)
    delta_row, delta_col = ACTIONS[action]
    
    new_row = row + delta_row
    new_col = col + delta_col
    
    # Check boundaries
    if 0 <= new_row < N and 0 <= new_col < N:
        return position_to_state(new_row, new_col)
    else:
        return state  # Stay in current state if out of bounds

# ============================================================================
# 3. VALUE ITERATION (BELLMAN EQUATION)
# ============================================================================

def value_iteration():
    """
    Perform value iteration until convergence.
    Returns the final value function.
    """
    global V
    
    iteration = 0
    
    print("Starting Value Iteration...")
    print(f"Grid Size: {N}x{N}")
    print(f"Gamma: {GAMMA}, Theta: {THETA}")
    print(f"Terminal State: {TERMINAL_STATE}\n")
    
    while True:
        delta = 0  # Track maximum change in value
        V_new = V.copy()  # Create copy for synchronous updates
        
        # Update value for each non-terminal state
        for state in range(NUM_STATES):
            if state == TERMINAL_STATE:
                V_new[state] = 0  # Terminal state value is always 0
                continue
            
            # Calculate value using Bellman equation
            # V(s) = (1/|A|) * sum_a [R + gamma * V(s')]
            action_values = []
            
            for action_name in ACTIONS:
                next_state = get_next_state(state, action_name)
                
                # Bellman update: reward + discounted next state value
                value = MOVE_REWARD + GAMMA * V[next_state]
                action_values.append(value)
            
            # Under uniform random policy, average all action values
            V_new[state] = np.mean(action_values)
            
            # Track maximum change for convergence check
            delta = max(delta, abs(V_new[state] - V[state]))
        
        # Update value function
        V = V_new
        iteration += 1
        
        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: max delta = {delta:.6f}")
        
        # Check convergence
        if delta < THETA:
            print(f"\nConverged after {iteration} iterations!")
            print(f"Final max delta: {delta:.10f}\n")
            break
    
    return V

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run value iteration
    final_V = value_iteration()
    
    # Display results
    print("=" * 60)
    print("FINAL VALUE FUNCTION")
    print("=" * 60)
    
    # Reshape to grid for visualization
    V_grid = final_V.reshape((N, N))
    
    print("\nValue Function (as grid):")
    print(V_grid)
    
    print("\n\nFormatted output:")
    print(np.array2string(V_grid, precision=8, suppress_small=True))
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("Each value represents the expected cumulative reward")
    print("(negative of expected steps) to reach the goal from that state")
    print("under a uniform random policy.")
    print(f"\nState 0 (top-left): {V_grid[0, 0]:.2f} steps expected")
    print(f"State 15 (bottom-right/goal): {V_grid[3, 3]:.2f} (terminal)")
    
    # Display state numbers for reference
    print("\n" + "=" * 60)
    print("STATE NUMBERING (for reference)")
    print("=" * 60)
    state_grid = np.arange(NUM_STATES).reshape((N, N))
    print(state_grid)