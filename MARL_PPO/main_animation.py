'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Animation -------------------

This script is a testing script that tests trained agents in "Trained_models" on specified environments with obstacles
with the collaboration of the obstacle avoidance agent.

Animation is carried out thanks to matplotlib.animation
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

plt.rcParams['text.usetex'] = True

# Initialize different parameters for testing
args = load_parameters_testing("input_args_testing.txt")
N_agents = int(args['N_agents'])
max_steps = 200
args['max_episode_length'] = max_steps

# Create agent headings
cols = []
for idx in range(N_agents):
    cols.append("UAV n. " + str(idx + 1))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
markers = ["D", "o", "X", "v", "^", "<", ">", "1", "2", "3", "4"]


print("---------- Testing procedure MARL PPO ----------")
print("Number of agents: " + str(int(args['N_agents'])))


# Load the actor model by selecting version
action_version = "final.h5"
actor = tf.keras.models.load_model('Trained_models/N_' + str(N_agents) + '_agents/actor_model_' + action_version)
#actor = tf.keras.models.load_model('Trained_models/N_' + str(4) + '_agents/actor_model_' + action_version)
# Load the actor model of the obstacle avoidance
oa_actor = tf.keras.models.load_model('Trained_models/OA_Models/Actor_final_PPO_09.h5')
num_actions = int(args['num_actions'])
num_obstacle_dirs = int(args['num_obstacle_dirs'])
num_inputs_oa_actor = num_actions + num_obstacle_dirs

# Create directory for saving all statistics of testing procedure
dir_test_stats = ["Statistics_Test", "Trajectories_Test"]
create_directories(dir_test_stats)

# Initialization of the environment
env = Env(args)

# State list
pos_map_list = []
cov_map_list = []
rec_map_list = []

# Reset the environment for testing
map_path = "../Maps/Case_maps/Map_2.txt"
#map_path = "../Maps/Testing_Maps/Testing_map_290"
states, positions, cells, trajectories, statistics = env.reset_test(args, random_choice = False, path = map_path, mut_dist=5)
ind_increases = []
cov_percs = []
#pos_map_list.append(states[0][:,:,0])
pos_map_list.append(create_pos_map_anim(cells))
cov_map_list.append(states[0][:, :,1])
rec_map_list.append(env.recmap)
cov_percs.append(coverage_percentage(env.covmap))

episode_return = 0
plotmaps = 0

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
    if plotmaps == 1:
        for id, state in enumerate(states):
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(state[:, :, 0])
            axs[1].imshow(state[:, :, 1], cmap="Greys", vmin=0, vmax=1)
            plt.savefig("CI/Fig_" + str(0) + "_" + str(t) + "_" + str(id) + ".png")
            plt.close()

    statistics.append(stats)
    episode_return += np.sum(rewards)

    for id_agent in range(N_agents):
        trajectories[id_agent].append(new_cells[id_agent])

    # Update the observation state
    states = new_states.copy()
    cells = new_cells.copy()
    #pos_map_list.append(np.abs(states[0][:, :, 0]))
    pos_map_list.append(create_pos_map_anim(cells))
    cov_map_list.append(states[0][:, :, 1])
    rec_map_list.append(env.recmap)

    if done:
        print("Performed " +str(env.nsteps) + " steps")
        print("Number of collisions: ", env.UAV_collisions)
        ind_increases = np.array(ind_increases)
        ind_increases_df = pd.DataFrame(ind_increases, columns = cols)
        print(ind_increases_df)
        break

'''
fig, axs = plt.subplots(2, 2, figsize = (10, 10))
# First image is the position map, second image is the coverage map
pos_map_fig = axs[0,0].imshow(pos_map_list[0], cmap = "YlOrRd", vmin = 0, vmax = 1, animated = True)
cov_map_fig = axs[0,1].imshow(cov_map_list[0], cmap = 'Oranges', animated = True)
rec_map_fig = axs[1,0].imshow(rec_map_list[0], cmap = 'GnBu', vmin = 0, vmax = 1, animated = True)
cov_per_fig,  = axs[1,1].plot([], [], linewidth = 2)
axs[1,1].grid()
axs[1,1].set_ylim(0, 100)
axs[1,1].set_xlim(0, env.nsteps)

axs[0, 0].set_title("Position map")
axs[0, 1].set_title("Coverage map")
axs[1, 0].set_title("Reconstructed map")
axs[1, 1].set_title("Coverage percentage")


axs[0, 0].get_xaxis().set_visible(False)
axs[0, 0].get_yaxis().set_visible(False)
axs[0, 1].get_xaxis().set_visible(False)
axs[0, 1].get_yaxis().set_visible(False)
axs[1, 0].get_xaxis().set_visible(False)
axs[1, 0].get_yaxis().set_visible(False)
step_text = axs[0, 0].text(0.02, 1.1, '', transform = axs[0, 0].transAxes)

def init():
    pos_map_fig.set_array(np.zeros((100, 100)))
    cov_map_fig.set_array(np.zeros((100, 100)))
    rec_map_fig.set_array(np.zeros((100, 100)))
    step_text.set_text('')
    cov_per_fig.set_data([], [])

    return [pos_map_fig], [cov_map_fig], [rec_map_fig], step_text, cov_per_fig,

def animate(i):
    global pos_map_list, cov_map_list, rec_map_list, cov_percs
    pos_map_fig.set_array(pos_map_list[i])
    cov_map_fig.set_array(cov_map_list[i])
    rec_map_fig.set_array(rec_map_list[i])
    cov_per_fig.set_data(np.arange(i), cov_percs[:i])
    step_text.set_text('Step = ' + str(i) + ", Coverage = " + str(round(cov_percs[i])) + "%")
    return [pos_map_fig], [cov_map_fig], [rec_map_fig], step_text, cov_per_fig,

ani = animation.FuncAnimation(fig, animate, frames = len(pos_map_list), interval = 100, blit = False, init_func = init, repeat = False )
ani.save(dir_test_stats[0]+"/Animation_" + str(N_agents)+'_agents.gif', dpi = 300)
'''

# Only the reconstruction animation
fig, ax = plt.subplots()
ims = []

for i in range(t):
    artist = []
    artist.append(plt.imshow(rec_map_list[i], cmap = 'GnBu', vmin = 0, vmax = 1.5, animated = True))
    for id_ag, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        # plt.scatter(trajectory[0, 1], trajectory[0, 0], marker = 'o')
        # axs[k_plotted].plot(trajectory[:, 1], trajectory[:, 0], linestyle=linestyle_list[id_ag])
        # axs[k_plotted].scatter(trajectory[:, 1], trajectory[:, 0], marker='o', s = 5)
        artist.append(plt.scatter(trajectory[i, 1], trajectory[i, 0], marker=markers[id_ag], color = colors[id_ag], s=50, animated = True))
        artist.append(plt.text(2, -10, 'Step = ' + str(i) + ", Coverage = " + str(round(cov_percs[i])) + "%"))
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False, left=False, labelbottom=False)
    if i == 0:
        artist.append(ax.imshow(rec_map_list[i], cmap = 'GnBu', vmin = 0, vmax = 1.5)) # show an initial one first)
        for id_ag, trajectory in enumerate(trajectories):
            trajectory = np.array(trajectory)
            # plt.scatter(trajectory[0, 1], trajectory[0, 0], marker = 'o')
            # axs[k_plotted].plot(trajectory[:, 1], trajectory[:, 0], linestyle=linestyle_list[id_ag])
            # axs[k_plotted].scatter(trajectory[:, 1], trajectory[:, 0], marker='o', s = 5)
            artist.append(
                plt.scatter(trajectory[i, 1], trajectory[i, 0], marker=markers[id_ag], color=colors[id_ag], s=50,
                            animated=True))
            artist.append(plt.text(2, -10, 'Step = ' + str(i) + ", Coverage = " + str(round(cov_percs[i])) + "\%"))
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False, left=False,
                           labelbottom=False)
    ims.append(artist)

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat = False)
ani.save(dir_test_stats[0]+"/Animation_" + str(N_agents)+'_agents_2.gif', dpi = 300)

# Compute EMA on the coverage per step and gaussian smoothing
ema = ind_increases_df.ewm(alpha = 0.05).mean()/ind_increases_df.max()
plot_trajectory_test(trajectories, env.recmap, 0)
ax1 = ema.plot(lw = 2, colormap = 'jet')
ax2 = ax1.twinx()
ax2.plot(cov_percs, label = "Coverage percentages")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Coverage per step")
ax2.set_ylabel("Map coverage (\%)")
ax1.set_title("Coverage N = "+ str(N_agents))
ax2.axhline(y = args['coverage_threshold']*100, color = 'y', linestyle = '--', lw = 0.5, label = "Coverage Threshold = "+str(round(args['coverage_threshold']*100)) + " %")
plt.grid()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=N_agents, fancybox=True, shadow=True)
#ax2.legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.1), ncol = 2, fancybox = True, shadow = True)
gau_smooth = ind_increases_df.rolling(window=int(env.nsteps/3), win_type='gaussian', center=True).mean(std=10)
gau_smooth.plot(lw = 2, colormap = 'jet')
#plt.savefig("Statistics_Test/Coverage_per_step.png")
plt.show()