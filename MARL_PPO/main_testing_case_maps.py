'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Test - case maps -------------------

Testing program allows to test the different trained agents according to the number of UAVs and specific training
hyperparameters, in collaboration to the obstacle avoidance agents trained using PPO and DQN.

Testing procedure works as follows:
- load the different hyperparameters
- load the obstacle avoidance agent (to be run on each UAV independently) - common to all simulations
- for each number of agent desired
    - load corresponding model
    - for each map in the testing maps
        - perform simulation and save all testing data

- write a .json file with all statistics

This program performs testing on the case maps in ../Maps/Case_maps
'''
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from scipy.io import savemat
from env_class import Env
from agent_function import *
import copy
from useful_functions import *
import pandas as pd
import json

# Use LaTeX fonts for plots
plt.rcParams['text.usetex'] = True

# Create directory for saving all statistics of testing procedure
dir_test_stats = ["Statistics_Test", "Trajectories_Test"]
create_directories(dir_test_stats)

# Initialize different parameters for testing
args = load_parameters_testing("input_args_testing.txt")

print("---------- Testing MARL PPO ----------")

# Initialize dictionary for statistics
stats_dict = {'N_agents': [], 'avg_distance': [], 'std_distance': [], 'std_pos': [], 'min_dist': [],
              'cov_50_steps': [], 'cov_100_steps': [], 'cov_200_steps': [], 'covered': [], 'n_steps': [],
              'map_occ': [], 'mean_travelled_dist' : []}

# Load the actor model of the obstacle avoidance
oa_actor = tf.keras.models.load_model('Trained_models/OA_Models/Actor_final_PPO_ver2.h5')
num_actions = int(args['num_actions'])
num_obstacle_dirs = int(args['num_obstacle_dirs'])
num_inputs_oa_actor = num_actions + num_obstacle_dirs

# Define the different combinations of agents to test the maps
N_agents_list = [2, 3, 4, 5, 6, 8, 10]
max_episode_length_list = 2000*np.ones((len(N_agents_list),))
coverage_threshold_list = [0.95, 0.97, 0.99, 0.99, 0.99, 0.99, 0.99]

for id_N, N_agents in enumerate(N_agents_list):
    args['coverage_threshold'] = coverage_threshold_list[id_N]
    args['max_episode_length'] = max_episode_length_list[id_N]
    max_steps = int(max_episode_length_list[id_N])
    args['N_agents'] = N_agents
    # Load the actor model from its current location
    trained_agent = "Trained_models/N_" + str(N_agents) + "_agents/actor_model_final.h5"
    actor = tf.keras.models.load_model(trained_agent)

    # Initialization of the environment
    env = Env(args)

    for id_map in range(5):
        print("N = "+str(N_agents), ", Map n. " + str(id_map))
        # Reset the environment for testing
        map_path = "../Maps/Case_maps/Map_"+str(id_map)+".txt"
        states, positions, cells, trajectories, statistics = env.reset_test(args, random_choice = False, path = map_path)
        ind_increases = []
        cov_percs = []

        episode_return = 0

        '''
        Statistics lists for each episode: 
        - std_pos_list              contains position standard deviation 
        - std_distances_list        contains distances standard deviation
        - avg_distances_list        contains average distances among UAVs 
        - min_distance_list         contains minimum distances among UAVs during simulation
        '''
        std_pos_list = []
        std_distances_list = []
        avg_distances_list = []
        min_distance_list = []
        distance_travelled_list = np.zeros((N_agents,))

        # Cycle and test the efficiency of the actor until no convergence is reached
        for t in range(max_steps):
            # Compute agents' actions
            agents_actions = []
            for id_agent in range(N_agents):
                logits, action = sample_action(tf.reshape(states[id_agent], (1, env.sz1, env.sz2, 2)), actor)
                agents_actions.append(action)
            new_states, new_cells, rewards, done, info, ind_inc, stats = env.step_test(cells, agents_actions, oa_actor, num_inputs_oa_actor)
            ind_increases.append(ind_inc)
            cov_percs.append(coverage_percentage(env.covmap)*100)

            statistics.append(stats)
            episode_return += np.sum(rewards)

            for id_agent in range(N_agents):
                trajectories[id_agent].append(new_cells[id_agent])
                distance_travelled_list[id_agent] += np.linalg.norm(new_cells[id_agent, :] - cells[id_agent, :])

            # Update the observation state
            states = new_states.copy()
            cells = new_cells.copy()

            # Add data to statistics
            std_pos_list.append(stats[0])
            std_distances_list.append(stats[1])
            avg_distances_list.append(stats[2])
            min_distance_list.append(stats[3])

            if done:
                print("Performed " +str(env.nsteps) + " steps")
                print("Number of collisions: ", env.UAV_collisions)
                ind_increases = np.array(ind_increases)
                break

        # Compute average distribution statistics
        mean_std_pos = np.mean(np.array(std_pos_list))
        mean_std_distance = np.mean(np.array(std_distances_list))
        mean_avg_distance = np.mean(np.array(avg_distances_list))
        mean_min_distance = np.mean(np.array(min_distance_list))
        mean_travelled_distance = np.mean(distance_travelled_list)

        # Compute coverage at different stops
        len_ep = len(cov_percs)
        if len_ep > 49:
            cov_50_steps = cov_percs[49]
        else:
            cov_50_steps = cov_percs[-1]
        if len_ep > 99:
            cov_100_steps = cov_percs[99]
        else:
            cov_100_steps = cov_percs[-1]
        if len_ep > 199:
            cov_200_steps = cov_percs[199]
        else:
            cov_200_steps = cov_percs[-1]

        ep_covered = True if info == 1 else False

        # Append all data to dictionary of statistics
        stats_dict['N_agents'].append(N_agents)
        stats_dict['avg_distance'].append(mean_avg_distance)
        stats_dict['std_distance'].append(mean_std_distance)
        stats_dict['min_dist'].append(mean_min_distance)
        stats_dict['std_pos'].append(mean_std_pos)
        stats_dict['cov_50_steps'].append(cov_50_steps)
        stats_dict['cov_100_steps'].append(cov_100_steps)
        stats_dict['cov_200_steps'].append(cov_200_steps)
        stats_dict['covered'].append(ep_covered)
        stats_dict['n_steps'].append(env.nsteps)
        stats_dict['map_occ'].append(
            (np.count_nonzero(env.map) - int(env.sz1 * 2 + env.sz2 * 2 - 4)) / (
                np.product(env.map.shape)) * 100)
        stats_dict['mean_travelled_dist'].append(mean_travelled_distance)

savemat(dir_test_stats[0]+"/Testing_data.mat", stats_dict)
with open(dir_test_stats[0]+"/Testing_data.json", 'w') as f:
    json.dump(stats_dict, f)
