'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Useful functions  -------------------

This script is intended as a collection of useful functions used during training
'''

import numpy as np
import os
import shutil
from keras.utils.vis_utils import plot_model
from tensorflow.keras import layers
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt

# Function to convert an array to an array of integers
def cast_to_integer(a):
    return a.astype(int)

# Function to load parameters from a file and check them
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
        return args
    except FileNotFoundError:
        raise FileNotFoundError("Input file "+filename+" not found")

# Function to create directories
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

# Function to create the neural networks
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

# Function to obtain occupied cell from position in [0, 1] x [0, 1]
def compute_UAV_cell(pos, sz1, sz2):
    # UAV_pos (2,) numpy array with positions in range [0, 1]
    # sz1 and sz2 => Number of cells along the 2 dimensions
    pos = np.squeeze(pos)
    # I must have that indices will be in [0, sz1-1]
    cells = cast_to_integer((np.round(pos * np.array([sz1-1, sz2-1]))))
    return cells

# Function to obtain current position in [0, 1], [0, 1] from occupied cells
def compute_UAV_pos(indices, sz1, sz2):
    # indices (2,) numpy array
    sz = np.array([sz1-1, sz2-1])   # Because maximum index is can be sz1-1 and/or sz2-1
    pos = indices/sz
    return pos

# Function to compute the increase of coverage between one map and another
def increase_coverage(old_map, new_map):
    increase = np.count_nonzero(new_map) - np.count_nonzero(old_map)    # Computes the increment in
    return increase

def from_position_to_axes(pos, sz1, sz2):
    '''
    Function used for plotting purposes with MATPLOTLIB IMSHOW function
    '''
    pos_image = np.array([pos[1]*sz2, pos[0]*sz1])
    return pos_image

# Function to randomly choose a training map
def pick_training_map(N):
    return np.random.choice(N)

# Function to compute coverage percentage of a map
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

def plot_trajectory(trajectory, rec_map, n_ep):
    trajectory = np.array(trajectory)
    plt.figure()
    plt.imshow(rec_map, cmap = "Greys", vmin = 0, vmax = 1)
    plt.scatter(trajectory[0, 1], trajectory[0, 0], marker = 'o')
    plt.plot(trajectory[:, 1], trajectory[:, 0])
    plt.scatter(trajectory[:, 1], trajectory[:, 0], marker = 'x')
    plt.savefig("Training_Trajectories/Episode_"+str(n_ep)+".png")
    plt.close()