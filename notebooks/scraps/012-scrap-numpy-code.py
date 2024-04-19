import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import pdb

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
nodes = list(range(1, 11))
G.add_nodes_from(nodes)

# Add edges to the graph
edges = [
    (1, 2), (1, 3), (1, 4),
    (2, 5), (2, 6),
    (3, 7), (3, 8),
    (4, 9), (4, 10),
    (5, 7), (5, 9),
    (6, 8), (6, 10),
    (7, 9),
    (8, 10)
]
G.add_edges_from(edges)

# Set node attributes
node_colors = ['lightblue', 'lightgreen', 'orange', 'pink', 'yellow',
               'purple', 'red', 'brown', 'gray', 'cyan']
node_sizes = [800 if node in [1, 5, 10] else 500 for node in nodes]

# Set edge attributes
edge_colors = ['black' if edge in [(1, 2), (1, 3), (1, 4)] else 'gray' for edge in edges]
edge_widths = [2 if edge in [(1, 2), (1, 3), (1, 4)] else 1 for edge in edges]

# # Draw the graph
# pos = nx.spring_layout(G, seed=42)
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
# nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)
# nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# # Show the graph
# plt.axis('off')
# plt.tight_layout()
# plt.show()

import gym
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class GraphEnv(gym.Env):
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.num_nodes = len(self.nodes)
        self.observation_space = gym.spaces.Discrete(self.num_nodes)
        self.action_space = gym.spaces.Discrete(self.num_nodes)
        self.current_node = None
        self.end_node = None
        self._node_to_mapping = {node: i for i, node in enumerate(self.nodes)}
        self._mapping_to_node = {i: node for i, node in enumerate(self.nodes)}

    def node_to_mapping(self, node):
        return self._node_to_mapping[node]
    
    def mapping_to_node(self, mapping):
        return self._mapping_to_node[mapping]

    def reset(self, start_node=None, end_node=None):
        if start_node is None:
            self.current_node = np.random.choice(self.nodes)
        else:
            self.current_node = start_node
        if end_node is None:
            self.end_node = np.random.choice(self.nodes)
        else:
            self.end_node = end_node
        return np.array([self.current_node])

    def step(self, action):
        if action in self.graph.neighbors(self.current_node):
            self.current_node = action
            if self.current_node == self.end_node:
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False
        else:
            reward = -1.0
            done = False
        return np.array([self.current_node]), reward, done, {}

    def render(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=500, font_size=16, font_weight='bold')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.current_node], node_color='red', node_size=500)
        plt.axis('off')
        plt.show()


def generate_expert_trajectories(graph, num_trajectories):
    """
    Generates expert trajectories in a graph.

    Parameters:
    - graph (networkx.Graph): The graph representing the environment.
    - num_trajectories (int): The number of trajectories to generate.

    Returns:
    - expert_trajectories (list): A list of expert trajectories, where each trajectory is a list of (observation, action) tuples.
    """

    expert_trajectories = []
    for _ in range(num_trajectories):
        start_node = np.random.choice(list(graph.nodes()))
        end_node = np.random.choice(list(graph.nodes()))
        while start_node == end_node:
            end_node = np.random.choice(list(graph.nodes()))
        path = nx.shortest_path(graph, source=start_node, target=end_node)        
            
        trajectory = []
        for i in range(len(path) - 1):
            obs = np.array([path[i]])
            action = path[i + 1]
            trajectory.append((obs, action))
        expert_trajectories.append(trajectory)
        
    
    return expert_trajectories



# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
nodes = list(range(1, 11))
G.add_nodes_from(nodes)

# Add edges to the graph
edges = [
    (1, 2), (1, 3), (1, 4),
    (2, 5), (2, 6),
    (3, 7), (3, 8),
    (4, 9), (4, 10),
    (5, 7), (5, 9),
    (6, 8), (6, 10),
    (7, 9),
    (8, 10)
]
G.add_edges_from(edges)

# Create the GraphEnv instance
env = GraphEnv(G)

# Reset the environment
env.reset()

def compute_state_visitation_freq(env, policy, trajectories):
    state_visitation_freq = np.zeros(env.num_nodes)
    total_visits = 0

    for trajectory in trajectories:
        state = env.node_to_mapping(trajectory[0][0][0])  # Convert NumPy array to a tuple
        for _, action in trajectory:
            if action in env.graph.neighbors(env.mapping_to_node(state)):
                state_visitation_freq[state] += 1
                total_visits += 1
                state = env.node_to_mapping(action)
            else:
                break

    state_visitation_freq /= total_visits
    return state_visitation_freq

def compute_maximum_entropy_policy(env, reward_function, temperature):
    policy = {}
    
    for state in env.nodes:
        neighbors = list(env.graph.neighbors(state))
        q_values = np.array([reward_function[env.node_to_mapping(neighbor)] for neighbor in neighbors])
        probabilities = np.exp(q_values / temperature)
        probabilities /= np.sum(probabilities)
        policy[state] = np.random.choice(neighbors, p=probabilities)
    return policy

def update_reward_function(expert_state_visitation_freq, learner_state_visitation_freq, reward_function, learning_rate):
    grad = expert_state_visitation_freq - learner_state_visitation_freq
    reward_function += learning_rate * grad
    return reward_function

def maximum_entropy_irl(env, expert_trajectories, num_iterations, learning_rate, temperature):
    num_states = env.num_nodes
    reward_function = np.zeros(num_states)
    expert_state_visitation_freq = np.zeros(num_states)

    for trajectory in expert_trajectories:
        for state, _ in trajectory:
            state_index = env.node_to_mapping(state[0])
            expert_state_visitation_freq[state_index] += 1

    expert_state_visitation_freq /= len(expert_trajectories)

    for _ in range(num_iterations):
        policy = compute_maximum_entropy_policy(env, reward_function, temperature)
        learner_state_visitation_freq = compute_state_visitation_freq(env, policy, expert_trajectories)
        reward_function = update_reward_function(expert_state_visitation_freq, learner_state_visitation_freq, reward_function, learning_rate)

    return reward_function, expert_state_visitation_freq

# Example usage
num_trajectories = 100
expert_trajectories = generate_expert_trajectories(G, num_trajectories)

num_iterations = 100
learning_rate = 0.1
temperature = 1.0
reward_function, expert_state_visitation_freq = maximum_entropy_irl(env, expert_trajectories, num_iterations, learning_rate, temperature)

print(reward_function) 
print(expert_state_visitation_freq)

# Normalize expert state visitation frequency
expert_state_visitation_freq_normalized = expert_state_visitation_freq / np.sum(expert_state_visitation_freq)

# Normalize reward function
reward_function_normalized = reward_function / np.sum(reward_function)
# Plotting the arrays
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(reward_function_normalized)), reward_function_normalized, color='blue', label='Normalized Reward Function')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Normalized Reward Function')

plt.subplot(1, 2, 2)
plt.bar(range(len(expert_state_visitation_freq_normalized)), expert_state_visitation_freq_normalized, color='red', label='Normalized Expert State Visitation Frequency')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Normalized Expert State Visitation Frequency')

plt.tight_layout()
plt.show()


