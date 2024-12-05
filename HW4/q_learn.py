import numpy as np
import matplotlib.pyplot as plt
import csv

class GridWorld:
    def __init__(self, dimensions, obstacles, reward_map, start_position, goal_position):
        self.grid_dimensions = dimensions  # Grid dimensions as a tuple (rows, cols)
        self.states = [(row, col) for row in range(self.grid_dimensions[0]) for col in range(self.grid_dimensions[1])]
        self.obstacle_positions = obstacles
        self.reward_mapping = reward_map  # Dictionary with positions as keys and rewards as values
        self.start_position = start_position
        self.goal_position = goal_position
        self.action_set = ['up', 'down', 'left', 'right']
        self.reset()

    def reset(self):
        self.current_position = self.start_position
        return self.current_position

    def is_valid_state(self, position):
        if (0 <= position[0] < self.grid_dimensions[0] and 0 <= position[1] < self.grid_dimensions[1]) and (position not in self.obstacle_positions):
            return True
        return False

    def step(self, action):
        x, y = self.current_position
        if action == 'up':
            new_position = (x - 1, y)
        elif action == 'down':
            new_position = (x + 1, y)
        elif action == 'left':
            new_position = (x, y - 1)
        elif action == 'right':
            new_position = (x, y + 1)
        else:
            raise ValueError("Invalid action")

        if self.is_valid_state(new_position):
            self.current_position = new_position
        else:
            # Invalid move (obstacle or out of bounds), stay in the same position
            new_position = self.current_position

        reward = self.get_reward(new_position)

        # Check if the goal position is reached
        is_done = False
        if new_position == self.goal_position:
            is_done = True

        return new_position, reward, is_done

    def get_reward(self, position):
        # Implement negative step cost to encourage shorter paths
        step_penalty = -0.1
        reward = self.reward_mapping.get(position, 0.0) + step_penalty
        return reward  # Default reward is step penalty if position not in reward_mapping

    def get_available_actions(self, position):
        available_actions = []
        x, y = position
        for action in self.action_set:
            if action == 'up' and self.is_valid_state((x - 1, y)):
                available_actions.append(action)
            elif action == 'down' and self.is_valid_state((x + 1, y)):
                available_actions.append(action)
            elif action == 'left' and self.is_valid_state((x, y - 1)):
                available_actions.append(action)
            elif action == 'right' and self.is_valid_state((x, y + 1)):
                available_actions.append(action)
        return available_actions

def q_learning(environment, num_episodes, learning_rate, discount_factor, explore_strategy):
    # Initialize Q-table

    q_table = {s: {a: 0.0 for a in environment.get_available_actions(s)}
               for s in environment.states if environment.is_valid_state(s) and environment.get_available_actions(s)}

    prev_q_table = {s: {a: 0.0 for a in environment.get_available_actions(s)}
                    for s in q_table}

    episode_steps = []
    convergence_threshold = 1e-4  # Convergence threshold

    for ep in range(num_episodes):
        current_state = environment.reset()
        is_done = False
        step_count = 0
        max_value_change = 0

        while not is_done:
            valid_actions = environment.get_available_actions(current_state)

            # Choose action based on exploration strategy
            if explore_strategy['type'] == 'epsilon-greedy':
                epsilon = explore_strategy['value']
                if np.random.rand() < epsilon:
                    chosen_action = np.random.choice(valid_actions)
                else:
                    q_vals = [q_table[current_state][act] for act in valid_actions]
                    max_q_val = max(q_vals)
                    optimal_actions = [act for act in valid_actions if q_table[current_state][act] == max_q_val]
                    chosen_action = np.random.choice(optimal_actions)
            elif explore_strategy['type'] == 'boltzmann':
                temperature = explore_strategy['value']
                q_vals = np.array([q_table[current_state][act] for act in valid_actions])
                q_vals = q_vals - np.max(q_vals)  # Prevent underflow
                exp_q_vals = np.exp(q_vals / temperature)
                action_probs = exp_q_vals / np.sum(exp_q_vals)
                chosen_action = np.random.choice(valid_actions, p=action_probs)
            else:
                raise ValueError("Invalid exploration strategy")

            next_state, reward, is_done = environment.step(chosen_action)

            # Q-learning update
            next_valid_actions = environment.get_available_actions(next_state)
            if next_valid_actions:
                max_next_q_val = max([q_table[next_state][act] for act in next_valid_actions])
            else:
                max_next_q_val = 0.0

            target_value = reward + discount_factor * max_next_q_val
            td_error = target_value - q_table[current_state][chosen_action]
            q_table[current_state][chosen_action] += learning_rate * td_error

            # Record the maximum change in Q-values for convergence check
            value_change = abs(q_table[current_state][chosen_action] - prev_q_table[current_state][chosen_action])
            if value_change > max_value_change:
                max_value_change = value_change
            prev_q_table[current_state][chosen_action] = q_table[current_state][chosen_action]

            current_state = next_state
            step_count += 1

            # Optional step limit per episode to prevent infinite loops
            if step_count > 1000:
                break

        episode_steps.append(step_count)

        # Update exploration parameters if needed
        if explore_strategy['type'] == 'boltzmann':
            # Update temperature with exponential decay
            initial_temperature = explore_strategy['initial_value']
            decay_rate = explore_strategy['decay_rate']
            explore_strategy['value'] = initial_temperature * np.exp(-decay_rate * ep)

        # Convergence check based on maximum Q-value change
        if max_value_change < convergence_threshold:
            print(f"Converged after {ep + 1} episodes")
            break

    return q_table, episode_steps

def write_q_values_to_csv(q_table, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["State", "Action", "Q-Value"])
        
        # Write each state-action pair and its Q-value
        for state, actions in q_table.items():
            for action, q_value in actions.items():
                writer.writerow([state, action, q_value])
    
    print(f"Q-values have been written to {filename}.")

# Define GridWorld environment parameters
grid_dimensions = (10, 10)
obstacles = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8),
             (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)]
state_rewards = {(5, 5): 1.0, (3, 3): -1.0, (7, 3): -1.0, (4, 5): -1.0,
                 (4, 6): -1.0, (5, 6): -1.0, (7, 5): -1.0, (6, 8): -1.0,
                 (5, 8): -1.0, (7, 6): -1.0}
initial_state = (0, 0)
goal_state = (5, 5)

environment = GridWorld(grid_dimensions, obstacles, state_rewards, initial_state, goal_state)
learning_rate = 0.01  # Learning rate
discount_factor = 0.9  # Discount factor
number_of_episodes = 5000

# Experiment with ε-greedy policy
epsilon_values = [0.1, 0.2, 0.3]
for epsilon in epsilon_values:
    exploration_config = {'type': 'epsilon-greedy', 'value': epsilon}
    q_table, steps_per_episode = q_learning(environment, number_of_episodes, learning_rate, discount_factor, exploration_config)
    print(f"\nε-Greedy Exploration with ε={epsilon}")
    print(f"Episodes to convergence: {len(steps_per_episode)}")
    # Print Q-values for the initial state
    print(f"Converged Q-values for the initial state {initial_state}:")
    for available_action in environment.get_available_actions(initial_state):
        q_value = q_table[initial_state][available_action]
        print(f"Action {available_action}: Q-value {q_value:.4f}")
    # Plot episode lengths
    plt.plot(steps_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.show()
    write_q_values_to_csv(q_table, f"q_values{epsilon}.csv")

# Experiment with Boltzmann exploration
exploration_config = {
    'type': 'boltzmann',
    'value': 10.0,  # Initial temperature T0
    'initial_value': 10.0,  # For decay calculation
    'decay_rate': 0.1  # Decay rate k
}
q_table, steps_per_episode = q_learning(environment, number_of_episodes, learning_rate, discount_factor, exploration_config)
print("\nBoltzmann Exploration")
print(f"Episodes to convergence: {len(steps_per_episode)}")
# Print Q-values for the initial state
print(f"Converged Q-values for the initial state {initial_state}:")
for available_action in environment.get_available_actions(initial_state):
    q_value = q_table[initial_state][available_action]
    print(f"Action {available_action}: Q-value {q_value:.4f}")
# Plot episode lengths
plt.plot(steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.show()
write_q_values_to_csv(q_table, "q_values_boltzman.csv")
