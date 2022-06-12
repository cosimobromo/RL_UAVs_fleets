'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Animation images -------------------

This script is a testing script that tests trained agents in "Trained_models" on specified environments with obstacles
with the collaboration of the obstacle avoidance agent.

Its aim is to produce images for thesis and paper
'''
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from scipy.io import savemat
from matplotlib import animation
from env_class import Env
from agent_function import *
import copy
from useful_functions import *
import pandas as pd
import json

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Initialize different parameters for testing
args = load_parameters_testing("input_args_testing.txt")   # Contains general parameters for the environment
N_agents = int(args['N_agents'])                           # Number of agents to be simulated in the environment
max_steps = 2000                                           # Maximum number of steps (to avoid endless loops)
args['max_episode_length'] = max_steps

# Create agent headings for figures
cols = []
for idx in range(N_agents):
    cols.append("UAV n. " + str(idx + 1))


print("---------- Testing procedure MARL PPO ----------")
print("Number of agents: " + str(int(args['N_agents'])))

# Load the COVERAGE actor model by selecting version
action_version = "final.h5"
actor = tf.keras.models.load_model('Trained_models/N_' + str(N_agents) + '_agents/actor_model_' + action_version)
#actor = tf.keras.models.load_model('Trained_models/N_' + str(4) + '_agents/actor_model_' + action_version) # To use models with N != from what simulated

# Load the OA actor model
oa_actor = tf.keras.models.load_model('Trained_models/OA_Models/Actor_final_PPO_09.h5')
num_actions = int(args['num_actions'])
num_obstacle_dirs = int(args['num_obstacle_dirs'])
num_inputs_oa_actor = num_actions + num_obstacle_dirs
markers = ["D", "o", "X", "v", "^", "<", ">", "1", "2", "3", "4"]

# Create directory for saving all statistics of testing procedure
dir_test_stats = ["Statistics_Test", "Trajectories_Test"]
create_directories(dir_test_stats)

# Initialization of the environment
env = Env(args)

# State list
pos_map_list = []
cov_map_list = []
rec_map_list = []
min_dist_list = []
avg_dist_list = []

# Reset the environment for testing
map_path = "../Maps/Case_maps/Map_2.txt"
#map_path = "../Maps/Testing_Maps/Testing_map_290"
states, positions, cells, trajectories, statistics = env.reset_test(args, random_choice = False, path = map_path, mut_dist=5)

initial_positions = positions.copy()
ind_increases = []
cov_percs = []
#pos_map_list.append(states[0][:,:,0])
pos_map_list.append(create_pos_map_anim(cells))
cov_map_list.append(states[0][:, :, 1])
rec_map_list.append(env.recmap + create_pos_map_anim(cells))
cov_percs.append(coverage_percentage(env.covmap))

episode_return = 0
plotmaps = 0

t_plot_list = [10, 50, 100]
k_plotted = 0

fig, axs = plt.subplots(1,3, figsize = (20, 10))
# Cycle and test the efficiency of the actor until no convergence is reached
for t in range(max_steps):
    # Compute agents' actions
    agents_actions = []
    for id_agent in range(N_agents):
        logits, action = sample_action(tf.reshape(states[id_agent], (1, env.sz1, env.sz2, 2)), actor)
        agents_actions.append(action)
    new_states, new_cells, rewards, done, info, ind_inc, stats = env.step_test(cells, agents_actions, oa_actor, num_inputs_oa_actor)
    min_dist_list.append(stats[3])
    avg_dist_list.append(stats[2])
    ind_increases.append(ind_inc)
    cov_percs.append(coverage_percentage(env.covmap)*100)

    statistics.append(stats)
    episode_return += np.sum(rewards)

    for id_agent in range(N_agents):
        trajectories[id_agent].append(new_cells[id_agent])

    if t in t_plot_list:
        densely_dashdotted = (0, (3, 1, 1, 1))
        densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))
        densely_dashed = (0, (5, 1))
        linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', densely_dashdotted, densely_dashdotdotted, densely_dashed]
        axs[k_plotted].imshow(env.recmap, cmap="Blues", vmin=0, vmax=1.5)
        for id_ag, trajectory in enumerate(trajectories):
            trajectory = np.array(trajectory)
            # plt.scatter(trajectory[0, 1], trajectory[0, 0], marker = 'o')
            #axs[k_plotted].plot(trajectory[:, 1], trajectory[:, 0], linestyle=linestyle_list[id_ag])
            #axs[k_plotted].scatter(trajectory[:, 1], trajectory[:, 0], marker='o', s = 5)
            axs[k_plotted].scatter(trajectory[-1,1], trajectory[-1,0], marker = markers[id_ag], s = 50)
            axs[k_plotted].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axs[k_plotted].tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False, left=False,
                                   labelbottom=False)

        axs[k_plotted].text(0.02, 1.05,
                 'Step ' + str(t) + " Coverage: " + str(round(np.count_nonzero(env.recmap) / 100)) + str("\%"),
                 transform=axs[k_plotted].transAxes,
                 bbox=dict(fill=None, boxstyle='round'), fontsize=20)
        k_plotted += 1
        if k_plotted == 3:
            #plt.savefig("Statistics_Test/Trajectories.eps", format = "eps")
            plt.savefig("Statistics_Test/Maps_example.png")

    #plot_trajectory_test(trajectories, env.recmap, t)
    # Update the observation state
    states = new_states.copy()
    cells = new_cells.copy()
    pos_map_list.append(create_pos_map_anim(cells))
    cov_map_list.append(states[0][:, :, 1])
    rec_map_list.append(env.recmap + create_pos_map_anim(cells))

    if done:
        print("Performed " +str(env.nsteps) + " steps")
        print("Number of collisions: ", env.UAV_collisions)
        ind_increases = np.array(ind_increases)
        ind_increases_df = pd.DataFrame(ind_increases, columns = cols)
        print(ind_increases_df)
        break


'''
idxs = [20, 50, 100]

fig, axs = plt.subplots(1, 3, figsize = (30, 10))

for id_ax, idx in enumerate(idxs):
    axs[id_ax].imshow(rec_map_list[idx], cmap = "Blues", vmin = 0, vmax = 1)
    axs[id_ax].get_xaxis().set_visible(False)
    axs[id_ax].get_yaxis().set_visible(False)
    axs[id_ax].text(0.02, 1.05, 'Step ' + str(idx) + " Coverage: " + str(round(cov_percs[idx])) + str("\%"), transform = axs[id_ax].transAxes,
                    bbox=dict(fill=True, facecolor = 'deepskyblue', alpha = 0.8, boxstyle='round,pad=1'), fontsize = 20)

plt.savefig("Statistics_Test/Rec_maps_"+str(N_agents)+".eps", format = "eps")
'''

# Plot test trajectories
plot_trajectory_test(trajectories, env.recmap, 0)

# Compute EMA on the coverage per step and gaussian smoothing
ema = ind_increases_df.ewm(alpha = 0.05).mean()/ind_increases_df.max()
ax1 = ema.plot(lw = 2, colormap = 'jet')
ax2 = ax1.twinx()
ax2.plot(cov_percs, label = "Coverage percentages")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Coverage per step")
ax2.set_ylabel("Map coverage (\%)")
ax1.set_title("Coverage N = "+ str(N_agents))
ax2.axhline(y = args['coverage_threshold']*100, color = 'y', linestyle = '--', lw = 0.5, label = "Coverage Threshold = "+str(round(args['coverage_threshold']*100)) + " %")
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True)
#ax2.legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.1), ncol = 2, fancybox = True, shadow = True)
plt.savefig(dir_test_stats[0]+"/Coverage_per_step_exp.png")
gau_smooth = ind_increases_df.rolling(window=int(env.nsteps/3), win_type='gaussian', center=True).mean(std=10)
gau_smooth.plot(lw = 2, colormap = 'jet')
plt.savefig(dir_test_stats[0]+"/Coverage_per_step_gaus.png")

# Plot initial positions properly
fig, axs = plt.subplots()
axs.imshow(env.map, cmap = "Greys", vmin = 0, vmax = 1)
for id_ag in range(N_agents):
    cell = compute_UAV_cell(initial_positions[id_ag], env.sz1, env.sz2)
    axs.scatter(cell[1], cell[0], marker = markers[id_ag])
    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)
    plt.savefig(dir_test_stats[0]+"/Initial_positions.png")

# Plot cumulative coverage
densely_dashdotted = (0, (3, 1, 1, 1))
densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))
densely_dashed = (0, (5, 1))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', densely_dashdotted, densely_dashdotdotted, densely_dashed]
cum_sum_ind_inc = np.cumsum(ind_increases, axis = 0)
cum_sum = np.cumsum(cum_sum_ind_inc, axis = 1)/(np.product(env.map.shape))
fig, axs = plt.subplots(1,1)
for id_ag in range(N_agents):
    plt.plot(cum_sum[:, id_ag], color = 'black') #color = colors[id_ag])
    if id_ag == 0:
        axs.fill_between(np.arange(cum_sum.shape[0]), np.zeros((cum_sum.shape[0])), cum_sum[:,id_ag],  label = "UAV n. "+str(id_ag+1), color = colors[id_ag])
    else:
        axs.fill_between(np.arange(cum_sum.shape[0]), cum_sum[:, id_ag-1], cum_sum[:, id_ag],  label = "UAV n. "+str(id_ag+1),
                         color=colors[id_ag])

plt.xlabel("Episode step")
plt.ylabel(r"Individually covered areas $\displaystyle \Delta_i$")
plt.legend()
plt.xlim((0,t))
plt.ylim((0,1))
plt.savefig(dir_test_stats[0]+"/Cumulative_individual_coverage.png")

# Plot avg and minimum distance history
fig, axs = plt.subplots(1,1)
axs.plot(np.array(avg_dist_list), label = "Mutual Distance (average)")
axs.plot(np.array(min_dist_list), label = "Minimum distance")
axs.plot(np.array([0, t]), np.array([0.1,0.1]), '-.', label = "Selected threshold", color = "red")
plt.title("AMUD and AMID")
plt.xlabel("Step time")
plt.ylabel("Distance")
plt.grid()
plt.legend()
plt.savefig(dir_test_stats[0]+"/Distance_over_time.png")
plt.show()