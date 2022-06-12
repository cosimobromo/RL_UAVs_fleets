'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Useful functions  -------------------

This script is intended as a collection of useful functions:
- create_pos_map:           Create the position map, i.e. the first layer of the state that is the input of each agent
- create_pos_map:           Create a modified version of the position map for animation purposes
- cast_to_integer:          Convert an array to integer
- load_parameters:          Load parameters for training from a .txt file
- load_parameters_testing:  Load paramrters for testing from a .txt file
- create_directories:       Create the directories for saving all data and models (with error handling)
- create_networks:          Create the actor and critic networks
- compute_UAV_cell:         From position in [0,1]x[0,1] compute the occupied cell
- compute_UAV_pos:          From cells move to position in [0,1]x[0,1]
- increase_coverage:        Compute coverage increase providing old and new coverage map
- pick_training_map:        To randomly pick a training map in the interval (setting N = 1 -> No obstacle map is selected)
- plot_statistics:          Plot and save all the training statistics
- plot_trajectory:          Plot current episode trajectory during training
- plot_trajectory_test:     Plot current episode trajectory during testing
- compute_mutual_distances: Compute mutual distances among UAVs
- compute_distances_id:     Compute distances of all UAVs with respect to a specified one
'''

import numpy as np
import os
import shutil
from agent_function import *
from keras.utils.vis_utils import plot_model
from tensorflow.keras import layers
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt


def create_pos_map(pos, id_current, N_agents, sz1 = 100, sz2 = 100, kernel_size = 21):
    '''
    This function takes as input the positions of the different UAVs and creates the position matrix properly
    '''
    N = kernel_size
    k1d = signal.gaussian(N, std = 2.5).reshape(N, 1)
    kernel = np.outer(k1d, k1d)
    pos_map = np.zeros((sz1,sz2))

    # Current UAV position considers a positive kernel
    row = pos[id_current, 0]
    col = pos[id_current, 1]

    ll_k = 0
    rl_k = N - 1
    ul_k = 0
    bl_k = N - 1

    up_lim = max(row - N // 2, 0)
    bot_lim = min(row + N // 2 + 1, sz1 - 1)
    left_lim = max(col - N // 2, 0)
    right_lim = min(col + N // 2 + 1, sz2 - 1)

    if np.argmax(np.array([row - N // 2, 0])) == 1:
        ul_k = N // 2 - row

    if np.argmax(np.array([col - N // 2, 0])) == 1:
        ll_k = N // 2 - col

    if np.argmin(np.array([row + N // 2 + 1, sz1 - 1])) == 1:
        bl_k = N - (N // 2 + row + 1 - sz1 + 1) - 1

    if np.argmin(np.array([col + N // 2 + 1, sz2 - 1])) == 1:
        rl_k = N - (N // 2 + col + 1 - sz2 + 1) - 1

    pos_map[up_lim:bot_lim, left_lim:right_lim] += kernel[ul_k:bl_k + 1, ll_k:rl_k + 1]

    # For other UAVs position put a negative kernel
    for id_uav in range(0, pos.shape[0]):
        if id_uav != id_current:
            row = pos[id_uav, 0]
            col = pos[id_uav, 1]

            ll_k = 0
            rl_k = N - 1
            ul_k = 0
            bl_k = N - 1

            up_lim = max(row - N // 2, 0)
            bot_lim = min(row + N // 2 + 1, sz1 - 1)
            left_lim = max(col - N // 2, 0)
            right_lim = min(col + N // 2 + 1, sz2 - 1)

            if np.argmax(np.array([row - N // 2, 0])) == 1:
                ul_k = N // 2 - row

            if np.argmax(np.array([col - N // 2, 0])) == 1:
                ll_k = N // 2 - col

            if np.argmin(np.array([row + N // 2 + 1, sz1 - 1])) == 1:
                bl_k = N - (N // 2 + row + 1 - sz1 + 1) - 1

            if np.argmin(np.array([col + N // 2 + 1, sz2 - 1])) == 1:
                rl_k = N - (N // 2 + col + 1 - sz2 + 1) - 1

            pos_map[up_lim:bot_lim, left_lim:right_lim] -= kernel[ul_k:bl_k + 1, ll_k:rl_k + 1]

    pos_map = np.clip(pos_map, -np.ones((sz1, sz2)), np.ones((sz1,sz2)))
    return pos_map


def create_pos_map_anim(pos, sz1 = 100, sz2 = 100, kernel_size = 21):
    '''
    This function takes as input the positions of the different UAVs and creates the position matrix properly,
    for animation purposes
    '''
    N = kernel_size
    k1d = signal.gaussian(N, std = 2.5).reshape(N, 1)
    kernel = np.outer(k1d, k1d)
    pos_map = np.zeros((sz1,sz2))

    # Current UAV position considers a positive kernel
    N_agents = pos.shape[0]
    for id_agent in range(N_agents):

        row = pos[id_agent, 0]
        col = pos[id_agent, 1]

        ll_k = 0
        rl_k = N - 1
        ul_k = 0
        bl_k = N - 1

        up_lim = max(row - N // 2, 0)
        bot_lim = min(row + N // 2 + 1, sz1 - 1)
        left_lim = max(col - N // 2, 0)
        right_lim = min(col + N // 2 + 1, sz2 - 1)
        if np.argmax(np.array([row - N // 2, 0])) == 1:
            ul_k = N // 2 - row

        if np.argmax(np.array([col - N // 2, 0])) == 1:
            ll_k = N // 2 - col

        if np.argmin(np.array([row + N // 2 + 1, sz1 - 1])) == 1:
            bl_k = N - (N // 2 + row + 1 - sz1 + 1) - 1

        if np.argmin(np.array([col + N // 2 + 1, sz2 - 1])) == 1:
            rl_k = N - (N // 2 + col + 1 - sz2 + 1) - 1

        pos_map[up_lim:bot_lim, left_lim:right_lim] += kernel[ul_k:bl_k + 1, ll_k:rl_k + 1]
        pos_map = np.clip(pos_map, np.zeros((sz1, sz2)), 2*np.ones((sz1,sz2)))

    return pos_map


def cast_to_integer(a):
    return a.astype(int)


def load_parameters(filename):
    try:
        args = {}
        with open(filename) as f:
            for line in f:
                (key, val) = line.split()
                args[key] = np.float32(val)
        # Check that FOV_dim is odd
        if int(args['FOV_dim']) % 2 == 0:
            quit("FOV_dim is even => It must be odd!")
        # Check that gamma, tau and target_kl are in [0, 1]
        if args['gamma'] < 0 or args['gamma'] > 1:
            quit("Gamma must be in [0, 1]")
        if args['lam'] < 0 or args['lam'] > 1:
            quit("Lambda must be in [0, 1]")
        if args['target_kl'] < 0 or args['target_kl'] > 1:
            quit("Target kl must be in [0, 1]")
        return args
    except FileNotFoundError:
        raise FileNotFoundError("Input file "+filename+" not found")


def load_parameters_testing(filename):
    try:
        args = {}
        with open(filename) as f:
            for line in f:
                (key, val) = line.split()
                args[key] = np.float32(val)
        # Check that FOV_dim is odd
        if int(args['FOV_dim']) % 2 == 0:
            quit("FOV_dim is even => It must be odd!")
        return args
    except FileNotFoundError:
        raise FileNotFoundError("Input file "+filename+" not found")


def create_directories(dirs):
    '''
    Function to be used for directory creation purposes
    '''
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            shutil.rmtree(dir)
            os.mkdir(dir)


def create_networks(args):
    '''
    Function to create the actor and critic Neural Network

    Inputs      --->    args: containing the input parameters
    Outputs     --->    actor network
                        critic network
    '''

    actor = create_actor(args)
    actor.compile(optimizer="Adam")
    critic = create_critic(args)
    critic.compile(optimizer="Adam")

    if int(args['plot_models']) == 1:      # Only once it is needed to plot the models
        plot_model(actor, to_file="NN_models"+'/Actor_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(critic, to_file="NN_models" + '/Critic_plot.png', show_shapes=True, show_layer_names=True)

    return actor, critic


def compute_UAV_cell(pos, sz1, sz2):
    # UAV_pos (2,) numpy array with positions in range [0, 1]
    # sz1 and sz2 => Number of cells along the 2 dimensions
    pos = np.squeeze(pos)
    # I must have that indices will be in [0, sz1-1]
    cells = cast_to_integer((np.round(pos * np.array([sz1-1, sz2-1]))))
    return cells


def compute_UAV_pos(indices, sz1, sz2):
    # indices (2,) numpy array
    sz = np.array([sz1-1, sz2-1])   # Because maximum index is can be sz1-1 and/or sz2-1
    pos = indices/sz
    return pos


def increase_coverage(old_map, new_map):
    increase = np.count_nonzero(new_map) - np.count_nonzero(old_map)    # Computes the increment in
    return increase


def pick_training_map(N):
    return np.random.choice(N)


def coverage_percentage(cov_map):
    sizes = cov_map.shape
    return np.count_nonzero(cov_map)/np.product(sizes)


def compute_mutual_distances(pos, threshold):
    '''
    This function allows to compute all the mutual distances among UAVs, given their respective positions
    '''

    bad_agents_ids = []

    N_agents = pos.shape[0]
    distances = np.zeros((int(N_agents*(N_agents-1)/2),))
    k = 0
    for i in range(N_agents):
        for j in range(i+1, N_agents):
            distances[k] = np.linalg.norm(pos[i, :]-pos[j, :])
            if distances[k] <= threshold:
                bad_agents_ids.append(i)
                bad_agents_ids.append(j)
            k+=1

    return distances, bad_agents_ids


def compute_distances_id(pos, threshold, id_ag):
    N_agents = pos.shape[0]
    distances = np.zeros((N_agents-1,))
    bad_agent = False
    k = 0
    for i in range(N_agents):
        if i != id_ag:
            distances[k] = np.linalg.norm(pos[i, :]-pos[id_ag, :])
            k = k+1

    if np.any(distances <= threshold):
        bad_agent = True

    return distances, bad_agent


def plot_trajectory(trajectories, rec_map, n_ep):
    plt.figure()
    plt.imshow(rec_map, cmap = "Greys", vmin = 0, vmax = 5)
    for trajectory in trajectories:
        trajectory = np.array(trajectory)
        plt.scatter(trajectory[0, 1], trajectory[0, 0], marker = 'o')
        plt.plot(trajectory[:, 1], trajectory[:, 0])
        plt.scatter(trajectory[:, 1], trajectory[:, 0], marker = 'x')
    plt.savefig("Training_Trajectories/Episode_"+str(n_ep)+".png")
    plt.close()


def plot_trajectory_test(trajectories, rec_map, n_ep):
    densely_dashdotted = (0, (3, 1, 1, 1))
    densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))
    densely_dashed = (0, (5, 1))
    linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', densely_dashdotted, densely_dashdotdotted, densely_dashed, 'solid', 'dotted', 'dashed']
    fig, axs = plt.subplots(figsize = (10, 6))
    plt.imshow(rec_map, cmap = "Greys", vmin = 0, vmax = 5)
    for id_ag, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        # Save the trajectory as a .csv file
        np.savetxt("Statistics_Test/Traj_"+str(id_ag+1)+".csv", trajectory, delimiter=",")
        #plt.scatter(trajectory[0, 1], trajectory[0, 0], marker = 'o')
        plt.plot(trajectory[:, 1], trajectory[:, 0], linestyle = linestyle_list[id_ag])
        plt.scatter(trajectory[:, 1], trajectory[:, 0], marker = 'o', s = 5)

    plt.text(0.02, 1.02, 'Step ' + str(n_ep) + " Coverage: " + str(round(np.count_nonzero(rec_map)/100)) + str("\%"),
                    transform=axs.transAxes,
                    bbox=dict(fill=None,  boxstyle='round'), fontsize=10)
    plt.savefig("Statistics_Test/Episode_"+str(n_ep)+".png")
    plt.close()


def plot_statistics(dic, path):
    dic_to_plot = dic.copy()
    n_ep = dic_to_plot['n_ep']
    n_epochs = dic_to_plot['n_epochs']
    del dic_to_plot['args']
    del dic_to_plot['n_ep']
    del dic_to_plot['n_epochs']

    episodes = np.arange(0, n_ep) + 1
    epochs = np.arange(0, n_epochs) + 1

    # Coverage plotting as function of episodes
    plt.figure()
    plt.plot(episodes, dic_to_plot['coverage'], '--', linewidth = 0.5, label = "Coverage per episode")
    plt.plot(episodes, dic_to_plot['ma_coverage'], linewidth = 2, label = "Moving average")
    plt.grid()
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Coverage percentage")
    plt.savefig(path+"/Coverage.png")

    # Step number plotting as function of episodes
    plt.figure()
    plt.plot(episodes, dic_to_plot['episode_steps'], '--', linewidth=0.5, label="Steps per episode")
    plt.plot(episodes, dic_to_plot['ma_episode_steps'], linewidth=2, label="Moving average of steps")
    plt.grid()
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.savefig(path + "/Episodic_steps.png")

    # Episode returns
    plt.figure()
    plt.plot(episodes, dic_to_plot['episode_return'], '--', linewidth=0.5, label="Episodic return")
    plt.plot(episodes, dic_to_plot['ma_episode_return'], linewidth=2, label="Moving average of episodic return")
    plt.grid()
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.savefig(path + "/Episodic_returns.png")

    # Epoch returns
    plt.figure()
    plt.plot(epochs, dic_to_plot['epoch_return'], '--', linewidth=0.5, label="Epoch return")
    plt.plot(epochs, dic_to_plot['ma_epoch_return'], linewidth=2, label="Moving average of epoch return")
    plt.grid()
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Returns")
    plt.savefig(path + "/Epoch_returns.png")

    # Epoch avg returns
    plt.figure()
    plt.plot(epochs, dic_to_plot['epoch_avg_return'], '--', linewidth=0.5, label="Average epoch return")
    plt.plot(epochs, dic_to_plot['ma_epoch_avg_return'], linewidth=2, label="Moving average of average epoch return")
    plt.grid()
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Returns")
    plt.savefig(path + "/Epoch_avg_returns.png")

    # Episodic mutual collisions
    plt.figure()
    plt.plot(episodes, dic_to_plot['episodic_mutual_collisions'], '--', linewidth=0.5, label="Episodic collisions among UAVs ")
    plt.plot(episodes, dic_to_plot['ma_episodic_mutual_collisions'], linewidth=2, label="Moving average of episodic mutual collisions among UAVs")
    plt.grid()
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Collisions")
    plt.savefig(path + "/Episodic_mutual_collisions.png")

    # Episodic statistics
    ep_stats = np.array(dic_to_plot['episodic_statistics'])

    fig, axs = plt.subplots(2,2)
    fig.suptitle("Distance statistics - average per episode")
    col_names = ['Position Standard Deviation', 'Distances standard deviation', 'Average distances', 'Minimum distance']
    k = 0
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(episodes, ep_stats[:,k])
            axs[i, j].set_xlabel('Episode')
            axs[i, j].set_ylabel(col_names[k])
            axs[i, j].set_title(col_names[k])
            k += 1
    plt.savefig(path + "/Position_statistics.png")