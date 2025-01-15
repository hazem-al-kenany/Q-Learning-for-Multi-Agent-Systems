import random
import matplotlib.pyplot as plt
import csv

class Agent:
    def __init__(self, grid_size, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        self.grid_size = grid_size
        self.epsilon = epsilon  #probability of exploration
        self.learning_rate = learning_rate  #how much new information overrides the old q-value
        self.discount_factor = discount_factor  #how much future rewards are valued over immediate rewards
        self.q_table = {}  #stores the q-values for state-action pairs
        self.position = (0, 0)  #agent's current position in the grid

    def initialize_q_table(self): #initializes q-values for every possible state-action pair
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.q_table[(x, y)] = {"up": 0, "down": 0, "left": 0, "right": 0}

    def select_action(self): #chooses an action based on epsilon-greedy strategy
        if random.uniform(0, 1) < self.epsilon:  #explores randomly with probability epsilon
            return random.choice(["up", "down", "left", "right"])
        else:  #exploits the best known action based on the q-table
            state = self.position
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state): #updates q-value for a state-action pair using the q-learning update formula
        max_future_q = max(self.q_table[next_state].values())  #maximum q-value for the next state
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * max_future_q - self.q_table[state][action]
        )

    def move(self, action): #moves the agent based on the chosen action (while staying within grid boundaries)
        x, y = self.position
        if action == "up" and x > 0:
            x -= 1
        elif action == "down" and x < self.grid_size - 1:
            x += 1
        elif action == "left" and y > 0:
            y -= 1
        elif action == "right" and y < self.grid_size - 1:
            y += 1
        self.position = (x, y)


class Environment:
    def __init__(self, grid_size, obstacles, goal):
        self.grid_size = grid_size  #size of the grid (10x10)
        self.obstacles = obstacles  #positions of obstacles that the agent cannot pass through
        self.goal = goal  #the target position the agent must reach (bottom right corner)

    def is_valid_position(self, position): #checks if a position is valid (not an obstacle or out of bounds)
        x, y = position
        if position in self.obstacles or x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return False
        return True

    def get_reward(self, position): #returns the reward for the agent's current position
        if position == self.goal:
            return 20 #+20 reward for reaching the goal
        return -1  #-1 penalty for every step

def run_single_agent_simulation(grid_size, obstacles, goal, episodes): #simulation setup
    env = Environment(grid_size, obstacles, goal)  #initializes environment
    agent = Agent(grid_size)  #creates an agent object
    agent.initialize_q_table()  #initializes the q-table for the agent

    results = []  #stores results for each episode

    for episode in range(episodes):
        agent.position = (0, 0)  #resets the agent's position to the start
        total_reward = 0  #accumulates rewards for the episode
        steps = 0  #counts the number of steps in the episode

        while True:
            state = agent.position  #current state of the agent
            action = agent.select_action()  #selects an action using epsilon-greedy

            agent.move(action)  #moves the agent based on the chosen action
            if not env.is_valid_position(agent.position):
                agent.position = state  #undo the move if it leads to an invalid position

            reward = env.get_reward(agent.position)  #gets the reward for the new position
            total_reward += reward  #updates total reward
            steps += 1  #increments step counter

            if agent.position == goal:  #ends the episode if the agent reaches the goal
                break

            agent.update_q_value(state, action, reward, agent.position)  #updates the q-table

        results.append((episode + 1, total_reward, steps))  #logs the episode's results

    return results


def run_multi_agent_simulation(grid_size, obstacles, goal, episodes, num_agents=4):
    env = Environment(grid_size, obstacles, goal)  #initializes the environment
    agents = [Agent(grid_size) for _ in range(num_agents)]  #creates multiple agents
    for agent in agents:
        agent.initialize_q_table()  #initializes q-tables for all agents

    results = []  #stores results for each episode

    for episode in range(episodes):
        for agent in agents:
            agent.position = (0, 0)  #resets all agents' positions to the start

        total_rewards = 0  #accumulates rewards for all agents
        steps = 0  #counts the total number of steps across all agents

        for agent in agents:
            while True:
                state = agent.position  #current state of the agent
                action = agent.select_action()  #selects an action using epsilon-greedy

                agent.move(action)  #moves the agent based on the chosen action
                if not env.is_valid_position(agent.position):
                    agent.position = state  #undo the move if it leads to an invalid position

                reward = env.get_reward(agent.position)  #gets the reward for the new position
                total_rewards += reward  #updates total rewards
                steps += 1  #increments step counter

                if agent.position == goal:  #ends the agent's episode if it reaches the goal
                    break

                agent.update_q_value(state, action, reward, agent.position)  #updates the q-table

        results.append((episode + 1, total_rewards, steps))  #logs the episode's results

    return results


def save_results_to_csv(filename, data):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)  #creates a csv writer object
        writer.writerow(["Episode", "Total Reward", "Steps"])  #writes the header row
        for row in data:
            writer.writerow(row)  #writes each episode's results to the file


def plot_learning_curves(single_data, multi_data):
    plt.figure(figsize=(10, 5))  #creates a figure with a specific size

    #single agent plot
    plt.subplot(1, 2, 1)
    episodes_single = [row[0] for row in single_data]  #x-axis: episode numbers
    total_rewards_single = [row[1] for row in single_data]  #y-axis: total rewards
    plt.plot(episodes_single, total_rewards_single)  #plots the single-agent learning curve
    plt.title("Single Agent Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Rewards")

    #multi-agent plot
    plt.subplot(1, 2, 2)
    episodes_multi = [row[0] for row in multi_data] #x-axis: episode numbers
    total_rewards_multi = [row[1] for row in multi_data]  #y-axis: total rewards
    plt.plot(episodes_multi, total_rewards_multi)  #plots the multi-agent learning curve
    plt.title("Multi-Agent Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Rewards")

    plt.tight_layout()  #adjusts layout to prevent overlap
    plt.show()  #displays the plots


if __name__ == "__main__":
    grid_size = 10  #size of grid environment
    obstacles = [(2, 2), (3, 3), (4, 4), (5, 5)]  #positions of obstacles
    goal = (9, 9)  #target goal position

    #single agent simulation
    single_agent_results = run_single_agent_simulation(grid_size, obstacles, goal, 1000) #change number of simulations/episodes from here
    save_results_to_csv("single_agent_results.csv", single_agent_results)  #save results to the csv file

    #multi-agent simulation
    multi_agent_results = run_multi_agent_simulation(grid_size, obstacles, goal, 1000) #change number of simulations/episodes from here
    save_results_to_csv("multi_agent_results.csv", multi_agent_results)  #save results to the csv file

    #plot both learning curves
    plot_learning_curves(single_agent_results, multi_agent_results)  #plots the learning curves