# Assignment1: GridWorld Value Iteration

A Python implementation of Value Iteration algorithm for a 4x4 GridWorld environment using the Bellman equation.

## Problem Description

The agent navigates a 4×4 grid from the **top-left corner (state 0)** to the **bottom-right corner (state 15)** using a uniform random policy.

### Environment Setup
- **Grid Size**: 4×4 (16 states)
- **Start State**: 0 (top-left)
- **Terminal State**: 15 (bottom-right)
- **Actions**: Up, Down, Left, Right (equal probability: 0.25 each)
- **Rewards**: -1 for each move, 0 for terminal state
- **Discount Factor (γ)**: 1.0 (no discounting)
- **Convergence Threshold (θ)**: 1e-4

### State Layout
```
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15 (GOAL)
```

## Algorithm

The implementation uses **Value Iteration** with the Bellman equation:

```
V(s) = (1/|A|) × Σ_a [R(s,a,s') + γ × V(s')]
```

Where:
- `V(s)` = value of state s
- `|A|` = number of actions (4)
- `R(s,a,s')` = immediate reward (-1 for moves)
- `γ` = discount factor (1.0)
- `V(s')` = value of next state

## Implementation Steps

1. **Initialize** value function V(s) = 0 for all states
2. **Iterate** until convergence:
   - For each non-terminal state:
     - Calculate expected value over all actions
     - Update V(s) using Bellman equation
   - Check if max change < θ
3. **Converge** when maximum value change < 1e-4

## Requirements

```bash
pip install numpy
```

## Usage

### Run the Script
```bash
python gridworld_value_iteration.py
```

### As a Jupyter Notebook
```python
# Run all cells in gridworld_value_iteration.ipynb
jupyter notebook gridworld_value_iteration.ipynb
```

## Output

### Final Value Function

```
[[-59.42367735 -57.42387125 -54.2813141  -51.71012579]
 [-57.42387125 -54.56699476 -49.71029394 -45.13926711]
 [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
 [-51.71012579 -45.13926711 -29.99766609   0.        ]]
```

### Interpretation

Each value represents the **expected cumulative reward** (negative of expected steps) to reach the goal from that state under a uniform random policy:

- **State 0** (top-left): ~59.42 expected steps to goal
- **State 15** (bottom-right): 0 (terminal state)
- **States closer to goal**: Higher (less negative) values
- **Corner states**: More negative values (farther from goal)

The gradient shows that states closer to the terminal state have higher values, as expected.

## Algorithm Details

### Convergence
- Typically converges in **~50-100 iterations**
- Convergence criterion: `max|V_new(s) - V_old(s)| < 1e-4`

### Policy
- **Uniform Random Policy**: Each action has probability 0.25
- This is a **policy evaluation** problem (not optimization)
- Optimal policy would be deterministic towards the goal

## Mathematical Background

### Bellman Equation (Policy Evaluation)
```
V^π(s) = Σ_a π(a|s) × Σ_s' P(s'|s,a) × [R(s,a,s') + γ × V^π(s')]
```

For uniform random policy: `π(a|s) = 1/4` for all actions

### Transition Dynamics
- **Deterministic**: Each action leads to one specific next state
- **Boundary handling**: Actions that would leave grid keep agent in current state

## Extensions

Possible improvements:
1. **Policy Iteration**: Find optimal policy (not just evaluate random policy)
2. **Q-Learning**: Model-free reinforcement learning
3. **Obstacles**: Add blocked states to make navigation harder
4. **Stochastic Actions**: Add noise to action outcomes
5. **Variable Rewards**: Different rewards for different regions

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- David Silver's RL Course: [Lecture on Dynamic Programming](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

## License

MIT License - Feel free to use and modify!

## Author

Created as an educational implementation of Value Iteration for GridWorld environments.

---

# Assignment 2: DQN Self-Driving Car Navigation (V2 with Safety Override)

A Deep Q-Network (DQN) implementation for autonomous car navigation on city maps using reinforcement learning.

## Overview

This project implements a self-driving car that learns to navigate through a city map, avoiding obstacles and reaching multiple sequential targets using Deep Q-Learning.

**Version 2** includes a **Safety Override System** that uses sensor data directly to prevent crashes, rather than waiting for the neural network to learn obstacle avoidance.

### Features

- **DQN with Enhanced Architecture**: 6-layer neural network with 512-unit hidden layer
- **Safety Override System**: Prevents dangerous actions using real-time sensor data
- **Ray-casting Sensors**: 7 LIDAR-style sensors that check along the entire ray path
- **Wide FOV**: 150° field of view for better junction/obstacle detection
- **Multiple Sequential Targets**: Navigate A1 → A2 → A3 in order
- **Real-time Visualization**: Watch the car learn with reward charts and statistics
- **Prioritized Experience Replay**: Faster learning from successful episodes

## Version Comparison (V1 vs V2)

| Feature | V1 (citymap_assignment_old.py) | V2 (citymap_assignment.py) |
|---------|--------------------------------|----------------------------|
| Safety Override | None | Active crash prevention |
| Sensor Type | Endpoint check only | Ray-casting (full path) |
| Field of View | 90° | 150° |
| SENSOR_DIST | 100 px | 50 px |
| SPEED | 3 px/step | 2 px/step |
| Target Detection | 35 px radius | 25 px radius |
| Epsilon Min | 0.05 | 0.001 |

### Safety Override System

The V2 safety system (`safety_override()` method) actively prevents dangerous actions:

1. **Emergency Detection**: Monitors center sensors for imminent collisions
2. **Direction Assessment**: Calculates left vs right safety scores
3. **Action Override**: Replaces dangerous actions with safe alternatives
4. **Threshold-based**: Uses danger thresholds (15%, 30%, 50% of sensor range)

## Requirements

```bash
pip install -r requirements.txt
# Or manually:
pip install torch numpy PyQt6
```

## Quick Start

### 1. Run the Application

```bash
cd ass16
python3 citymap_assignment.py
```

### 2. Setup Navigation

1. **Left-click** on map to place car starting position
2. **Left-click** 3 times to place targets (A1, A2, A3)
3. **Right-click** to confirm target placement
4. Press **SPACE** to start training

### 3. Watch the Car Learn!

The car will initially move randomly, then gradually learn to:
- Stay on roads (bright pixels)
- Avoid obstacles (dark pixels)
- Navigate to all 3 targets in sequence

## File Structure

```
ass16/
├── citymap_assignment.py       # V2: Main application with Safety Override
├── citymap_assignment_old.py   # V1: Original implementation (for comparison)
├── maps/                       # City map images
│   ├── map0.jpg - map5.jpg
├── CHANGES.md                  # Documentation of all code changes
├── ASSIGNMENT_SOLUTION.md      # Answers to theory questions
├── TRAINING_GUIDE.md           # Training & video creation guide
├── verify_setup.py             # Setup verification script (for V1)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Parameters (V2)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| SENSOR_DIST | 50 | Sensor range in pixels |
| SENSOR_ANGLE | 15° | Angle between sensors (150° total FOV) |
| SPEED | 2 | Movement speed per step |
| TURN_SPEED | 8° | Normal turn angle |
| SHARP_TURN | 25° | Sharp turn angle |
| BATCH_SIZE | 64 | Training batch size |
| GAMMA | 0.95 | Future reward discount |
| LR | 0.001 | Learning rate |
| TAU | 0.005 | Target network update rate |
| epsilon | 1.0→0.001 | Exploration rate (decays) |

## Neural Network Architecture

```
Input (9) → FC(128) → ReLU → FC(256) → ReLU → FC(512) → ReLU → FC(256) → ReLU → FC(128) → ReLU → Output(5)
```

**Input**: 7 sensor values + angle to target + distance to target  
**Output**: Q-values for 5 actions (left, straight, right, sharp-left, sharp-right)

## Available Maps

Maps are located in the `maps/` folder:

| Map | Description |
|-----|-------------|
| map0.jpg | Basic city layout |
| map1.jpg | Realistic aerial city view |
| map2.jpg | Grid city with waterways |
| map3.jpg | Paris-style with river (default) |
| map4.jpg | Night aerial (stunning visuals) |
| map5.jpg | River city, organic layout |

## Documentation

- **[CHANGES.md](CHANGES.md)** - All modifications made to the original code
- **[ASSIGNMENT_SOLUTION.md](ASSIGNMENT_SOLUTION.md)** - Answers to theory questions
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training and video guide

## Troubleshooting

```bash
# Verify setup (for V1 parameters)
python3 verify_setup.py

# Check dependencies
pip install torch numpy PyQt6
```

## Video Demo

[YouTube Video Link - To Be Added After Recording]

## License

MIT License