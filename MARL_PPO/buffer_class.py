'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Buffer class --------------------------

This script contains:
- discounted_cumulative_sums function, used for generalised advantage estimation and return estimation
- Buffer class with methods:
    - Initialization
    - Store:                for storing each step of the simulation
    - Get:                  to get all data from the buffer for training purposes
    - Finish_trajectory:    for last value estimation and appending

'''

import numpy as np
import tensorflow as tf
import scipy


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Buffer:
    def __init__(self, args):
        # buffer_capacity:      length of the episode to store properly
        self.buffer_capacity = int(args['max_length'])
        self.gamma, self.lam = args['gamma'], args['lam']

        num_actions = int(args['num_actions'])
        sz1 = int(args['sz1'])
        sz2 = int(args['sz2'])

        # State is 2 layer map of dimension sz1 x sz2
        self.observation_buffer = np.zeros((self.buffer_capacity, sz1, sz2, 2), dtype = np.float32)
        self.action_buffer = np.zeros(self.buffer_capacity, dtype=np.int32)
        self.advantage_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.return_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.value_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.logprobability_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]
        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )