'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Training of Obstacle avoidance using PPO  -------------------

This script is intended for the training procedure of the RL agent that, as function of the high level direction of
motion selected by the coverage agent and on the basis of obstacle detections in its neighbourhood, decides the best
direction to take to reach such position
'''

# Importing all needed libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from useful_functions import *
from env_class import Env
from scipy.io import savemat
import wandb
import scipy

# To choose whether to plot trajectories of training episodes
plot_trajectories = 1

# Directories for trajectories and Neural Network models
dirs = ["Trajectories", "NN_models"]
create_directories(dirs)                # From useful functions, used to create such directories with error handling

# Load training parameters
args = load_parameters("input_args.txt")

# wandb configuration parameters
wandb.init(project="Obstacle_Avoidance", entity="cosimobromo")
wandb.run.name = "PPO_Vers"
wandb.config = args

# Read maximum steps per episode from args
max_steps_per_episode = int(args['max_episode_length'])

# Create env object using Env class
env = Env(args)

num_actions = int(args['num_actions'])                  # Number of possible actions
num_obstacle_dirs = int(args['num_obstacle_dirs'])      # Number of directions containing obstacle presence

# Discounted cumulative sums: used for computing rewards-to-go and advantage estimates
def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    '''
    Buffer for storing trajectories of each training epoch, containing:
    - observation buffer (state)
    - action buffer
    - advantage buffer
    - reward buffer
    - return buffer
    - value buffer
    - logprobability buffer
    '''
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        # To keep track of the current buffer length and position in storing trajectories
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
        path_slice = slice(self.trajectory_start_index, self.pointer)         # From initial point of the trajectory to the pointer (final recorded step)
        rewards = np.append(self.reward_buffer[path_slice], last_value)       # Append last step reward
        values = np.append(self.value_buffer[path_slice], last_value)         # Append last step value

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]         # Define the deltas
        # Delta(t) = reward(t) + gamma*V(t+1) - V(t)                          # Like the advantage defined
        # Add elements in the advantage buffer and in the return buffer (return != reward)
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

def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    # Apply softmax function
    logprobabilities_all = tf.nn.log_softmax(logits)

    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )                       # Compute log probability corresponding to the specific chosen action
    return logprobability

# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl

# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

'''
This function creates the Neural Network for action logits outputs (as many output elements as the number of possible actions)
'''
def create_actor():
    inputs = layers.Input(shape=(int(num_actions+num_obstacle_dirs),))
    out = layers.Dense(512, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(inputs)
    out = layers.Dense(256, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(out)
    out = layers.Dense(128, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(out)
    out = layers.Dense(64, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(out)
    out = layers.Dense(64, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(out)
    outputs = layers.Dense(9, kernel_initializer = tf.keras.initializers.random_uniform(minval = -3e-3, maxval = 3e-3))(out)
    model = keras.Model(inputs = inputs, outputs = outputs)
    return model

'''
This function creates the Neural Network allowing to approximate the value function in a given state
'''
def create_critic():
    inputs = layers.Input(shape=(int(num_actions+num_obstacle_dirs),))
    out = layers.Dense(512, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(inputs)
    out = layers.Dense(256, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(out)
    out = layers.Dense(128, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(out)
    out = layers.Dense(64, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(out)
    out = layers.Dense(64, activation="relu", kernel_initializer = tf.keras.initializers.HeNormal())(out)
    outputs = layers.Dense(1, kernel_initializer = tf.keras.initializers.random_uniform(minval = -3e-3, maxval = 3e-3))(out)
    model = keras.Model(inputs = inputs, outputs = outputs)
    return model

# Total number of elements in the state
observation_dimensions = num_obstacle_dirs + num_actions

# Hyperparameters of the PPO algorithm
steps_per_epoch = 1500
epochs = 1000
gamma = 0.9
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 20
train_value_iterations = 20
model_saving_rate = 50
episode_plotting_rate = 100
lam = 0.97
target_kl = 0.01

# Initialization of the buffer class
buffer = Buffer(observation_dimensions, steps_per_epoch, gamma, lam)

# Initialize the actor and the critic as keras models
actor = create_actor()
critic = create_critic()

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize lists for statistics analysis
episode_reward_list = []
correct_behaviour_list = []
info_list = []

# Initialize the observation, episode return and episode length, and reset the environment as well
state, pos, cell = env.reset(args)
episode_return = 0
episode_length = 0
trajectory = []
trajectory.append(cell)     # Save first cell position in the trajectory list
n_ep = 0                    # Counter for the number of episodes

for epoch in range(epochs):
    sum_return = 0
    sum_length = 0
    # During the epoch, everything is performed by sampling and taking actions
    for t in range(steps_per_epoch):
        logits, action = sample_action(tf.reshape(state, (1, observation_dimensions)))      # Sample an action
        new_state, new_cell, reward, done, info = env.step(state, cell, action)             # Take the step according to the current action
        new_state = np.array(new_state)
        episode_return += reward
        episode_length += 1

        # Get value and log probability of the action
        value_t = critic(tf.reshape(state, (1, observation_dimensions)))            # Estimate the value using the critic
        logprobability_t = logprobabilities(logits, action)                         # Compute log probability of the current action using the given state

        # Store in the buffer
        buffer.store(state, action, reward, value_t, logprobability_t)

        state = new_state.copy()
        cell = new_cell.copy()
        trajectory.append(cell)

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(state.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            n_ep += 1
            if terminal:
                info_list.append(info)
                episode_reward_list.append(episode_return)
                correct_behaviour_list.append(episode_return/env.nsteps)
                if plot_trajectories == 1 and n_ep % episode_plotting_rate == 0:
                    trajectory = np.array(trajectory)
                    plt.figure(n_ep + 1)
                    plt.imshow(env.map, cmap="Greys", vmin=0, vmax=5)
                    plt.scatter(trajectory[0, 1], trajectory[0, 0], marker='o')
                    plt.plot(trajectory[:, 1], trajectory[:, 0], marker='x')
                    plt.savefig(dirs[0] + "/Ep_" + str(n_ep + 1) + ".png")
                    plt.close()
                wandb.log({'info': info, 'episodic_reward': episode_return, 'correct_behaviour': float(episode_return/env.nsteps)})
            state, pos, cell = env.reset(args)
            trajectory = []
            trajectory.append(cell)
            episode_return = 0
            episode_length = 0

    # Before moving to the next epoch, perform training
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)

    # Every <model_saving_rate> epochs update the current policy
    if epoch % model_saving_rate == 0:
        actor.save(dirs[1]+"/Actor_epoch_"+str(epoch)+".h5")

# Save final actor at the end of the training procedure
actor.save("Actor_final.h5")
dir_to_save = {'episodic_reward': episode_reward_list, 'info': info_list, 'correct_behave': correct_behaviour_list}
savemat("Training_Stats.mat", dir_to_save)
