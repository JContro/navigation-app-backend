import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
)
from imitation.data import rollout
from imitation.rewards import reward_nets
import gymnasium as gym
from gymnasium import spaces

class GraphEnv(gym.Env):
    def __init__(self, graph, horizon=100):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.num_nodes = len(self.nodes)
        self.observation_space = spaces.Discrete(self.num_nodes)
        self.action_space = spaces.Discrete(self.num_nodes)
        self.current_node = None
        self.end_node = None
        self._node_to_mapping = {node: i for i, node in enumerate(self.nodes)}
        self._mapping_to_node = {i: node for i, node in enumerate(self.nodes)}
        self.horizon = horizon
        self.steps = 0

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
        self.steps = 0
        return self.node_to_mapping(self.current_node)

    def step(self, action):
        action_node = self.mapping_to_node(action)
        if action_node in self.graph.neighbors(self.current_node):
            self.current_node = action_node
            if self.current_node == self.end_node:
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False
        else:
            reward = -1.0
            done = False
        self.steps += 1
        if self.steps >= self.horizon:
            done = True
        return self.node_to_mapping(self.current_node), reward, done, {}

    def render(self, mode='human'):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=500, font_size=16, font_weight='bold')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.current_node], node_color='red', node_size=500)
        plt.axis('off')
        plt.show()


class GraphEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.state_dim = env.num_nodes
        self.action_dim = env.num_nodes
        self.horizon = env.horizon
        self.transition_matrix = self._build_transition_matrix()
        self.reward_matrix = self._build_reward_matrix()

    def _build_transition_matrix(self):
        transition_matrix = np.zeros((self.state_dim, self.action_dim, self.state_dim))
        for state in range(self.state_dim):
            node = self.env.mapping_to_node(state)
            for action in range(self.action_dim):
                action_node = self.env.mapping_to_node(action)
                if action_node in self.env.graph.neighbors(node):
                    next_state = self.env.node_to_mapping(action_node)
                    transition_matrix[state, action, next_state] = 1.0
        print(f"Transition matrix shape: {transition_matrix.shape}")
        print(f"Transition matrix: {transition_matrix[0]}")
        return transition_matrix

    def _build_reward_matrix(self):
        reward_matrix = np.zeros((self.state_dim, self.action_dim))
        for state in range(self.state_dim):
            node = self.env.mapping_to_node(state)
            for action in range(self.action_dim):
                action_node = self.env.mapping_to_node(action)
                if action_node in self.env.graph.neighbors(node):
                    if action_node == self.env.end_node:
                        reward_matrix[state, action] = 1.0
                    else:
                        reward_matrix[state, action] = 0.0
                else:
                    reward_matrix[state, action] = -1.0
        print(f"Reward matrix shape: {reward_matrix.shape}")
        print(f"Reward matrix: {reward_matrix[0]}")
        return reward_matrix.reshape(self.state_dim, self.action_dim, 1)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

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

env = GraphEnvWrapper(GraphEnv(G))
env.reset()

num_trajectories = 100
expert_trajectories = generate_expert_trajectories(G, num_trajectories)

rng = np.random.default_rng(0)

def state_env_creator():
    return GraphEnvWrapper(GraphEnv(G))

state_venv = DummyVecEnv([state_env_creator] * 4)

_, _, pi = mce_partition_fh(env)
_, om = mce_occupancy_measures(env, pi=pi)

reward_net = reward_nets.BasicRewardNet(
    env.observation_space,
    env.action_space,
    hid_sizes=[256],
    use_action=False,
    use_done=False,
    use_next_state=False,
)

mce_irl = MCEIRL(
    om,
    env,
    reward_net,
    log_interval=250,
    optimizer_kwargs={"lr": 0.01},
    rng=rng,
)

occ_measure = mce_irl.train()

imitation_trajs = rollout.generate_trajectories(
    policy=mce_irl.policy,
    venv=state_venv,
    sample_until=rollout.make_min_timesteps(5000),
    rng=rng,
)

print("Imitation stats: ", rollout.rollout_stats(imitation_trajs))