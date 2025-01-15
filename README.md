# Q-Learning-for-Multi-Agent-Systems

This project simulates a reinforcement learning environment where single or multiple agents navigate a grid while learning optimal paths to reach a goal. The agents utilize Q-learning with an epsilon-greedy strategy for balancing exploration and exploitation.

---

## Features

### Single-Agent Simulation
- **Environment**: A grid-based environment with obstacles and a defined goal.
- **Agent Behavior**:
  - Navigates using actions: `up`, `down`, `left`, and `right`.
  - Learns through Q-learning by updating Q-values for state-action pairs.
  - Balances exploration (random moves) and exploitation (optimal moves based on learned Q-values).
- **Performance Tracking**:
  - Total rewards and steps are logged per episode.

### Multi-Agent Simulation
- **Multiple Agents**: Independent agents learn concurrently in the same environment.
- **Interactions**:
  - Each agent follows its own Q-learning process.
  - Results are aggregated to evaluate the overall performance.
- **Performance Metrics**:
  - Cumulative rewards and total steps for all agents per episode.

### Data Export and Visualization
- Results are saved in CSV files for further analysis.
- Learning curves for single-agent and multi-agent scenarios are plotted to compare performance.

---

## Code Structure

### Classes

#### **Agent**
- **Attributes**:
  - `q_table`: Stores Q-values for state-action pairs.
  - `epsilon`, `learning_rate`, `discount_factor`: Parameters for Q-learning.
- **Methods**:
  - `initialize_q_table`: Initializes the Q-table for all grid states.
  - `select_action`: Chooses actions using epsilon-greedy strategy.
  - `update_q_value`: Updates Q-values using the Q-learning formula.
  - `move`: Updates the agent's position based on the chosen action.

#### **Environment**
- **Attributes**:
  - `grid_size`: Defines the grid dimensions.
  - `obstacles`: Defines positions of obstacles.
  - `goal`: Target position the agents aim to reach.
- **Methods**:
  - `is_valid_position`: Checks if a position is valid.
  - `get_reward`: Returns a reward based on the agent's position.

### Functions

#### Simulation
- `run_single_agent_simulation`: Executes episodes for a single agent.
- `run_multi_agent_simulation`: Executes episodes for multiple agents simultaneously.

#### Utilities
- `save_results_to_csv`: Saves simulation results to a CSV file.
- `plot_learning_curves`: Visualizes learning curves for single and multi-agent scenarios.

---

## How to Run

### Prerequisites
- Python 3.7 or higher.
- Install required libraries:
  ```bash
  pip install matplotlib
