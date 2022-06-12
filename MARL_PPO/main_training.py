'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Training -------------------

This script is used for training, using PPO algorithm, a fleet of a variable number of UAVs, to cooperate in covering
an unknown area.

In training procedure, no obstacle presence is considered
'''

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import savemat
from env_class import Env
from tensorflow import keras
from agent_function import *
import os
import wandb
import shutil
from useful_functions import *
from buffer_class import Buffer
import copy

# Initialize counter of time needed for training
start_time = time.time()

# Load the different parameters from the input file (check of consistency is performed automatically)
args = load_parameters("input_args.txt")
print(args)

# wandb parameter configuration
use_wandb = int(args['use_wandb'])
if use_wandb == 1:
    import wandb
    wandb.init(project="MARL_COV_PPO", entity="cosimobromo")

'''
Now use the function to create all the proper directories, and overwrite the previous one: 
- Statistics:                   For saving trajectories
- NN_models:                    For saving .h5 files containing NN models during training
- Training_Trajectories:        For saving trajectories of episodes during training
'''
dirs = ["Statistics", "NN_models", "Training_Trajectories"]
create_directories(dirs)

'''
Creation of the Neural Networks for the agent: 
- Actor network
- Critic network (providing an estimate of the value function) 
'''
actor, critic = create_networks(args)  # (Also plot a model of them if requested - 'plot_model' entry in input_args.txt)

'''
Definition of policy and value optimizers. Use Adam Optimizer with learning rates specificed in 
input_args.txt
'''
policy_optimizer = keras.optimizers.Adam(learning_rate=args['actor_lr'])
value_optimizer = keras.optimizers.Adam(learning_rate=args['critic_lr'])

'''
Environment initialization (using 'Env' class) 
'''
env = Env(args)
view_range = env.Fd  # For D* Lite Path Planning

'''
Training procedure parameters: 
- N_agents:             number of agents during the training procedure
- num_actions:          number of possible actions: 
                        - 4: possibility to move up/down/left/right
                        - 8: possibility to move up/down/left/right/up-right/up-left/down-right/down-left
- motion_amount         Number of cells which affect the motion
- save_maps:            1 if maps (like trajectories) have to be saved, 0 otherwise
- use_pp:               1 if using path planning to reach the goal, 0 otherwise
- swp:                  sliding window for plotting of statistics
- total_epochs          Total number of epochs for training (# of trajectories to collect and number of calling of training)
- N_training            Number of training maps to be used 
- coverage_threshold    Threshold for considering the map fully covered 
- N_trial               Incremental number of the trial (for wandb organization purposes when running on cluster) 
- model_saving_rate     Rate (in terms of epochs) at which saving models for backup purposes 
 
At the end, initialize wandb if use_wandb = 1 
'''
N_agents = int(args['N_agents'])
num_actions = int(args['num_actions'])
motion_amount = int(args['motion_amount'])
save_maps = int(args['save_maps'])
use_pp = int(args['use_pp'])
swp = int(args['sliding_window_plot'])
total_epochs = int(args['total_epochs'])
N_training = int(args['N_training_maps'])
coverage_threshold = args['coverage_threshold']
N_trial = int(args['N_trial'])
model_saving_rate = int(args['model_saving_rate'])
if use_wandb == 1:
    wandb.run.name = "Run_Trial_" + str(N_trial)
    wandb.config = args

'''
RL Algorithm hyperparameters: 
- train_policy_iterations       Number of epochs for training of the policy (for each trajectory) 
- train_value_iterations        Number of epochs for training of the value - critic (for each trajectory) 
- max_length                    Maximum length of trajectories to be collected for PPO training 
- max_episode_length            Maximum length of each episode
- clip_ratio                    Clipping ratio for PPO (epsilon)
- target_kl                     Target value for KL for premature stopping of training 
'''
train_policy_iterations = int(args['train_policy_iterations'])
train_value_iterations = int(args['train_value_iterations'])
max_length = int(args['max_length'])
max_episode_length = int(args['max_episode_length'])
clip_ratio = args['clip_ratio']
target_kl = args['target_kl']

''' 
Buffers initialization 
'''
buffers = []
for id_agent in range(N_agents):
    buffers.append(Buffer(args))

'''
Initialize the lists to keep track of different statistics. 

Episode related lists: 
- info_list:                            contains information about episode ending causes
- coverage_list:                        contains information about coverage at the end of each episode
- ma_coverage_list:                     moving average of coverage list with a sliding window defined by variable 'swp'
- episode_steps_list:                   contains the number of steps per episode
- ma_episode_steps_list                 contains the moving average of the number of steps per episode
- episode_return_list                   contains the episodic return 
- ma_episode_return_list                contains the moving average of the episodic return 
- episodic_cov_per_step_list            contains, for each episode, the coverage attained at each step
- episodic_mutual_collisions_list       contains, for each episode, the number of overall collisions 
- ma_episodic_mutual_collisions_list    moving average of episodic mutual collisions list with sliding window 'swp'
- episodic_statistics_list              contains statistics of distance over episodes 

Epoch related lists: 
- epoch_return_list:                    contains the return of the overall epoch 
- ma_epoch_return_list:                 contains the moving average of epoch_return_list with a sliding window defined by variable 'swp'
- epoch_avg_return_list:                contains the return of the overall epoch divided by the number of episodes in that epoch 
- ma_epoch_avg_return_list:             contains the moving average of epoch_avg_return_list with a sliding window defined by variable 'swp'
'''
info_list = []
coverage_list = []
ma_coverage_list = []
episode_steps_list = []
ma_episode_steps_list = []
episode_return_list = []
ma_episode_return_list = []
episodic_cov_per_step_list = []
episodic_mutual_collisions_list = []
ma_episodic_mutual_collisions_list = []
episodic_statistics_list = []
epoch_return_list = []
ma_epoch_return_list = []
epoch_avg_return_list = []
ma_epoch_avg_return_list = []

'''
The overall training procedure involves: 
- reset of the environment and UAVs initialization at the beginning of each new episode 
- creation of the different inputs of the UAVs during training, storage in the replay buffer, application of the defined
  action and reward computation 
- At the end of each epoch: training neural network using the already built buffer replay 
- Statistics saving
'''

states, positions, cells, trajectories, statistics = env.reset(args)
episode_return = 0  # Overall episode return (of all agents in the environment)
n_ep = 0  # Episode number from the beginning of the training procedure

for epoch in range(total_epochs):
    # Initialize counters for the epoch
    epoch_return = 0
    sum_return = 0  # Overall returns in the epoch
    num_episodes = 0  # Number of episodes during the current epoch
    for t in range(max_length):

        ''' 
        At each time step, using the same actor model for each agent, compute logit and corresponding action for each 
        agent, and store such experience in the buffer before moving to the next step 
        '''
        agents_logits = []
        agents_actions = []

        '''
        Following code is for correct state checking: 
        
        for id, state in enumerate(states):
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(state[:, :, 0])
            axs[1].imshow(state[:, :, 1], cmap = "Greys", vmin = 0, vmax = 1)
            plt.savefig("CI/Fig_"+str(epoch)+"_"+str(t)+"_"+str(id)+".png")
            plt.close()
        '''
        for id_agent in range(N_agents):
            logits, action = sample_action(tf.reshape(states[id_agent], (1, env.sz1, env.sz2, 2)), actor)
            agents_logits.append(logits)
            agents_actions.append(action)

        new_states, new_cells, rewards, done, info, stats = env.step(cells, agents_actions)
        statistics.append(stats)
        episode_return += np.sum(rewards)

        for id_agent in range(N_agents):
            trajectories[id_agent].append(new_cells[id_agent])
            value_t = critic(tf.reshape(states[id_agent], (1, env.sz1, env.sz2, 2)))
            logprobability_t = logprobabilities(agents_logits[id_agent], agents_actions[id_agent], num_actions)
            buffers[id_agent].store(states[id_agent], agents_actions[id_agent], rewards[id_agent], value_t,
                                    logprobability_t)

        # Update the observation state
        states = new_states.copy()
        cells = new_cells.copy()

        # Finish trajectory if reached to a terminal state or if finished the epoch time
        terminal = done
        if terminal or (t == max_length - 1):
            for id_agent in range(N_agents):
                last_value = 0 if done else critic(tf.reshape(states[id_agent], (1, env.sz1, env.sz2, 2)))
                buffers[id_agent].finish_trajectory(last_value)
            sum_return += episode_return
            num_episodes += 1
            if terminal:
                episode_return_list.append(episode_return)
                ma_episode_return_list.append(np.mean(episode_return_list[swp:]))
                coverage_list.append(coverage_percentage(env.covmap))
                ma_coverage_list.append(np.mean(coverage_list[swp:]))
                info_list.append(info)
                episode_steps_list.append(env.nsteps)
                ma_episode_steps_list.append(np.mean(episode_steps_list[swp:]))
                episodic_mutual_collisions_list.append(env.UAV_collisions)
                ma_episodic_mutual_collisions_list.append(np.mean(episodic_mutual_collisions_list[swp:]))
                episodic_cov_per_step_list.append(env.coverage_per_step_list)
                mean_stats = np.mean(np.array(statistics), axis=0)
                episodic_statistics_list.append(mean_stats)
            '''
            In wandb, log coverage at: 
            - 50
            - 100
            - 150
            - 200
            - 250
            - 300
            steps from the beginning of each new episode
            '''
            if use_wandb != 0 and terminal:
                wandb.log({'episode_return': episode_return, 'ma_episode_return': ma_episode_return_list[-1],
                           'coverage': coverage_list[-1], 'ma_coverage': ma_coverage_list[-1], 'info': info_list[-1],
                           'episode_steps': episode_steps_list[-1], 'ma_episode_steps': ma_episode_steps_list[-1],
                           'mean_pos_std': mean_stats[0], 'mean_dist_std': mean_stats[1],
                           'mean_avg_dist': mean_stats[2],
                           'mean_min_dist': mean_stats[3], 'mutual_collisions': episodic_mutual_collisions_list[-1],
                           'ma_mutual_collisions': ma_episodic_mutual_collisions_list[-1],
                           'cov_50_steps': env.coverage_per_step_list[49],
                           'cov_100_steps': env.coverage_per_step_list[99],
                           'cov_150_steps': env.coverage_per_step_list[149],
                           'cov_200_steps': env.coverage_per_step_list[199],
                           'cov_250_steps': env.coverage_per_step_list[249],
                           'cov_300_steps': env.coverage_per_step_list[299],
                           })
                n_ep += 1
                # Plot current episode trajectory
                # plot_trajectory(trajectories, env.recmap, n_ep)

            states, positions, cells, trajectories, statistics = env.reset(args)
            episode_return = 0

    '''
    For each agent's buffer, perform an update of the network 
    '''
    for id_agent in range(N_agents):
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffers[id_agent].get()

        # Update policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, actor,
                              policy_optimizer, clip_ratio, max_length, num_actions)
            if kl > 1.5 * target_kl:
                # Early stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer, critic, value_optimizer)

    # Print mean return and length for each epoch
    epoch_return = sum_return
    epoch_mean_return = sum_return / num_episodes
    epoch_return_list.append(epoch_return)
    ma_epoch_return_list.append(np.mean(epoch_return_list[swp:]))
    epoch_avg_return_list.append(epoch_mean_return)
    ma_epoch_avg_return_list.append(np.mean(epoch_avg_return_list[swp:]))

    print("Epoch: " + str(epoch + 1) + " Mean return: " + str(epoch_mean_return))

    if epoch % model_saving_rate == 0:
        # Save the trained models and statistics
        actor.save(dirs[1] + "/actor_model_epoch_" + str(epoch) + ".h5")
        critic.save(dirs[1] + "/critic_model_epoch_" + str(epoch) + ".h5")

        end_time = time.time()
        ex_time = round(end_time - start_time, 2)
        dic_to_mat = {
            'args': args, 'info': info_list, 'n_ep': n_ep, 'n_epochs': total_epochs, 'coverage': coverage_list,
            'ma_coverage': ma_coverage_list, 'episode_steps': episode_steps_list,
            'ma_episode_steps': ma_episode_steps_list, 'episode_return': episode_return_list,
            'ma_episode_return': ma_episode_return_list, 'epoch_return': epoch_return_list,
            'ma_epoch_return': ma_epoch_return_list, 'epoch_avg_return': epoch_avg_return_list,
            'ma_epoch_avg_return': ma_epoch_avg_return_list, 'episodic_cov_per_step': episodic_cov_per_step_list,
            'episodic_mutual_collisions': episodic_mutual_collisions_list,
            'ma_episodic_mutual_collisions': ma_episodic_mutual_collisions_list,
            'episodic_statistics': episodic_statistics_list, 'execution_time': ex_time
        }
        # Plot statistics and save in the correct folder
        statistics_path = dirs[0] + "/Statistics_"+str(epoch)+".mat"
        savemat(statistics_path, dic_to_mat)

    if use_wandb != 0:
        wandb.log({'epoch': epoch, 'epoch_return': epoch_return, 'epoch_mean_return': epoch_mean_return,
                   'ma_epoch_return': ma_epoch_return_list[-1], 'ma_epoch_avg_return': ma_epoch_avg_return_list[-1],
                   'coverage': coverage_list[-1], 'ma_coverage': ma_coverage_list[-1], 'info': info_list[-1]})

    '''
    Check stop conditions: 
    - if the coverage is fully attained with less than 300 steps for 50 episodes AND 
    - no mutual collisions are present for the last 10 episodes 
    
    steps_window = -50
    min_epoch = 1000
    if np.mean(episodic_mutual_collisions_list[swp:]) == 0 and not np.any(np.array(episode_steps_list[steps_window:]) == max_episode_length) and epoch >= min_epoch:
        break
    '''

# Save the trained models and statistics
actor.save(dirs[1] + "/actor_model_final.h5")
critic.save(dirs[1] + "/critic_model_final.h5")
end_time = time.time()
ex_time = round(end_time - start_time, 2)
dic_to_mat = {
    'args': args, 'info': info_list, 'n_ep': n_ep, 'n_epochs': epoch, 'coverage': coverage_list,
    'ma_coverage': ma_coverage_list, 'episode_steps': episode_steps_list,
    'ma_episode_steps': ma_episode_steps_list, 'episode_return': episode_return_list,
    'ma_episode_return': ma_episode_return_list, 'epoch_return': epoch_return_list,
    'ma_epoch_return': ma_epoch_return_list, 'epoch_avg_return': epoch_avg_return_list,
    'ma_epoch_avg_return': ma_epoch_avg_return_list, 'episodic_cov_per_step': episodic_cov_per_step_list,
    'episodic_mutual_collisions': episodic_mutual_collisions_list,
    'ma_episodic_mutual_collisions': ma_episodic_mutual_collisions_list,
    'episodic_statistics': episodic_statistics_list, 'execution_time': ex_time
}
# Plot statistics and save in the correct folder
statistics_path = dirs[0] + "/Statistics_final.mat"
savemat(statistics_path, dic_to_mat)
plot_statistics(dic_to_mat, dirs[0])

print("Execution took " + str(ex_time) + " seconds")