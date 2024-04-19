import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class GraphEnv:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.num_nodes = len(self.nodes)
        self.observation_space = self.num_nodes
        self.action_space = self.num_nodes
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
        return torch.tensor([self.current_node])

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
        return torch.tensor([self.current_node]), reward, done, {}

    def render(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=500, font_size=16, font_weight='bold')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.current_node], node_color='red', node_size=500)
        plt.axis('off')
        plt.show()


def generate_expert_trajectories(graph, num_trajectories):
    expert_trajectories = []
    for _ in range(num_trajectories):
        start_node = np.random.choice(list(graph.nodes()))
        end_node = np.random.choice(list(graph.nodes()))
        while start_node == end_node:
            end_node = np.random.choice(list(graph.nodes()))
        path = nx.shortest_path(graph, source=start_node, target=end_node)        
            
        trajectory = []
        for i in range(len(path) - 1):
            obs = torch.tensor([path[i]])
            action = path[i + 1]
            trajectory.append((obs, action))
        expert_trajectories.append(trajectory)
    
    return expert_trajectories


def compute_state_visitation_freq(env, policy, trajectories):
    state_visitation_freq = torch.zeros(env.num_nodes)
    total_visits = 0

    for trajectory in trajectories:
        state = env.node_to_mapping(trajectory[0][0].item())  # Convert tensor to a scalar
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
        q_values = torch.tensor([reward_function[env.node_to_mapping(neighbor)] for neighbor in neighbors])
        probabilities = torch.exp(q_values / temperature)
        probabilities /= torch.sum(probabilities)
        policy[state] = np.random.choice(neighbors, p=probabilities.numpy())
    return policy


def maximum_entropy_irl(env, expert_trajectories, num_iterations, learning_rate, temperature):
    num_states = env.num_nodes
    reward_function = torch.zeros(num_states, requires_grad=True)
    expert_state_visitation_freq = torch.zeros(num_states)

    for trajectory in expert_trajectories:
        for state, _ in trajectory:
            state_index = env.node_to_mapping(state.item())
            expert_state_visitation_freq[state_index] += 1

    expert_state_visitation_freq /= len(expert_trajectories)

    optimizer = optim.Adam([reward_function], lr=learning_rate)

    for _ in range(num_iterations):
        policy = compute_maximum_entropy_policy(env, reward_function, temperature)
        learner_state_visitation_freq = compute_state_visitation_freq(env, policy, expert_trajectories)
        
        optimizer.zero_grad()
        entropy = -torch.sum(learner_state_visitation_freq * torch.log(learner_state_visitation_freq + 1e-8))
        loss = torch.sum((expert_state_visitation_freq - learner_state_visitation_freq) * reward_function) - temperature * entropy
        loss.backward()
        optimizer.step()

    return reward_function.detach()

# Example usage
G = nx.Graph()
nodes = list(range(1, 11))
G.add_nodes_from(nodes)
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

env = GraphEnv(G)
env.reset()

num_trajectories = 100
expert_trajectories = generate_expert_trajectories(G, num_trajectories)

num_iterations = 100
learning_rate = 0.1
temperature = 1.0
reward_function = maximum_entropy_irl(env, expert_trajectories, num_iterations, learning_rate, temperature)

print(reward_function)