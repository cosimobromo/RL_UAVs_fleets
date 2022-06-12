'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Agent functions -------------------

This script is used to define the main agent functions to be used during training:
- logprobability:               Allows to compute, basing on the logits and on the action selected, the log probability
                                pi(a|s)
- create_actor:                 Returns the actor Convolutional Neural Network (CNN)
- create_critic:                Returns the critic Convolutional Neural Network (CNN)
- sample_action:                Feeds the actor network with the current state, and on the basis of the logits it
                                selects an action accordingly to the categorical distribution
- train_policy:                 Trains the policy network
- train_value_function:         Trains the critic network
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from useful_functions import *

def logprobabilities(logits, a, num_actions):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


def create_actor(args):
    '''
    The actor model receives as input a 2 layer image containing:
    - 1st layer:    Binary coverage map [0, 1]
    - 2nd layer:    Matrix indicating the current UAV and the other UAV's positions (using Gaussian Bells)

    The actor model provides as output 9 logits, indicating probability of 9 actions:
    - 0: do nothing
    - 1: move up
    - 2: move down
    - 3: move left
    - 4: move right
    - 5: move up left
    - 6: move up right
    - 7: move down left
    - 8: move down right
    '''
    # Number of possible actions
    N_actions = int(args['num_actions'])
    # Layers initialization
    init_kind = tf.keras.initializers.HeNormal()

    # Map size definition
    sz1 = int(args['sz1'])
    sz2 = int(args['sz2'])

    # Define the input observation map
    map_input = layers.Input(shape=(sz1, sz2, 2))
    out = layers.Conv2D(8, (5, 5), input_shape=(sz1, sz2, 2), strides=(3, 3), activation="relu", padding="valid")(
        map_input)
    out = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(out)
    out = layers.Conv2D(16, (3, 3), strides=(2, 2), activation="relu", padding="valid")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Flatten()(out)
    out = layers.Dropout(0.1)(out)
    out = layers.Dense(512, activation="relu", kernel_initializer=init_kind)(out)
    out = layers.Dense(256, activation = "relu", kernel_initializer = init_kind)(out)
    out = layers.Dense(128, activation = "relu", kernel_initializer = init_kind)(out)
    out = layers.Dense(64, activation = "relu", kernel_initializer = init_kind)(out)
    action_out = layers.Dense(N_actions, kernel_initializer=init_kind)(out)

    model = Model(map_input, action_out)

    return model


def create_critic(args):
    '''
    The critic model has as input the same state provided to the actor model and outputs a singleton, corresponding
    to the value function of such state
    '''

    # Layers initialization
    init_kind = tf.keras.initializers.HeNormal()

    # Map size definition
    sz1 = int(args['sz1'])
    sz2 = int(args['sz2'])

    # Batch size definition
    batch_sz = int(args['batch_size'])

    # Define the input observation map
    map_input = layers.Input(shape=(sz1, sz2, 2))
    out = layers.Conv2D(8, (5, 5), input_shape=(sz1, sz2, 2), strides=(3, 3), activation="relu", padding="valid")(
        map_input)
    out = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(out)
    out = layers.Conv2D(16, (3, 3), strides=(2, 2), activation="relu", padding="valid")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Flatten()(out)
    out = layers.Dense(512, activation="relu", kernel_initializer=init_kind)(out)
    out = layers.Dense(256, activation="relu", kernel_initializer=init_kind)(out)
    out = layers.Dense(128, activation="relu", kernel_initializer=init_kind)(out)
    out = layers.Dense(64, activation="relu", kernel_initializer=init_kind)(out)
    outputs = layers.Dense(1, kernel_initializer = init_kind)(out)                  # Value function V

    model = Model(map_input, outputs)
    return model


@tf.function
def sample_action(observation, actor_model):
    '''
    Compute logits using the actor model and select action according to the categorical distribution
    '''
    logits = actor_model(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis = 1)
    return logits, action


# Train the policy by maximizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, actor, policy_optimizer, clip_ratio, max_length, num_actions):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(tf.reshape(tf.convert_to_tensor(observation_buffer), (max_length, 100, 100, 2))), action_buffer, num_actions)
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
        - logprobabilities(actor(observation_buffer), action_buffer, num_actions)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer, critic, value_optimizer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

