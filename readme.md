# Reinforcement Learning based Strategic Exploration Algorithm for UAVs fleets

## Introduction
This repository contains the files and scripts developed during my Master's Thesis project in Mechatronic Engineering with title **"Reinforcement Learning Based Strategic Exploration Algorithm for UAVs fleets"** in a.y. 2021/2022 at Polytechnic of Turin (www.polito.it). 

## Project Description
The available code is implemented to solve the coverage planning problem of unknown environments through UAVs fleets, exploiting Reinforcement Learning (RL) based algorithms applied to multi-agent systems. Variable size fleets, consisting of 2 to 10 units, are trained concurrently in shared complex environments to maximize their individually explored areas while keeping well distributed, thus learning a mixed competitive-collaborative behaviour. Proximal Policy Optimization (PPO) algorithm is modified in order to address these multi-agent settings, exploiting all agents’ trajectories collected during training epochs to update a shared policy function, suitably approximated by means of a Convolutional Neural Network (CNN). Coverage planning and obstacle avoidance tasks are carried out by means of 2 trained RL-agents, running independently on each fleet unit acquiring shared and local information and processing input states to assigned waypoints. 

<p align="center">
	<img src="/images/Exp_example.png" alt="Exploration example" width="3000"/>
</p>

Trained models are tested in environments with variable complexity, either in 2D and 3D simulations. The former are performed without considering the effect of path planning methods between consequent waypoints, the latter involve ROS nodes interacting with worlds simulated in Gazebo and UAVs controlled by Flight Control Units implemented using PX4 Autopilot software (https://px4.io). 

<p align="center">
	<img src="/images/ROS_img.jpg" alt="Initial part of the exploration phase in ROS Gazebo" width="3000"/>
</p>

## Repository organization 
This repository is organized as follows: 
1. `Maps`: this folder contains the scripts used to generate random maps with different complexities by adding obstacles. 
	* `Training_Maps`: contains 2700 maps for agent training;
	* `Testing_Maps`: contains 300 maps for test; 
	* `Case_Maps`: contains reference maps for performance assessment; 
	* `Map_example.png`: example of a randomly generated map; 
	* `maps_creation.py`: class containing methods for maps creation; 
	* `distribution_maps.py`: scripts to plot distribution of obstacle occupancies in test maps; 
	* `main_single_map.py`: script for generation of a single map with specified characteristics;
	* `main_multiple_maps.py`: script for generation of the database of maps. 
2. `Obst_avoidance_PPO`: this folder contains the scripts used for training the *obstacle avoidance agent*: 
	* `main_training.py`: script for training the obstacle avoidance agent;
	* `env_class.py`: contains the environment class definition, with all methods needed to handle transitions, reset and reward computation;
	* `useful_function.py`: collection of functions used for parameters loading, plots and auxiliary computations
	* `input_args.txt`: collection of parameters for tuning the training process
	* `NN_models/Actor_final.h5`: .h5 model of the trained OA model
3. `MARL_PPO`: this folder contains the scripts used for training and testing the *coverage agent*: 
	* `main_training.py`: script for training the coverage agent;
	* `agent_function.py`: script for defining the CNN structures of actor and critic models, and to handle training phase (using Tensorflow); 
	* `buffer_class.py`: script to handle buffer reading, writing and GAE method;
	* `env_class.py`: contains the environment class definition, with all methods needed to handle transitions, reset and reward computation;
	* `useful_function.py`: collection of functions used for parameters loading, plots and auxiliary computations;
	* `input_args.txt`: collection of parameters for tuning the training process;
	* `input_args_testing.txt`: collection of parameters for tuning the test process;
	* `main_animation.py`: script to test and animate the exploration process;
	* `main_animation_images.py`: script to test an exploration process and plot statistics and performance metrics;
	* `main_testing.py`: script to test the model on *all* test maps;
	* `main_testing_case_maps.py`: script to test the model on the maps selected as *reference*;
	* `Trained_models/N_..._agents/Actor_final.h5`: .h5 model of the trained coverage agent model for a fleet of dimension ...
4. `catkin_ws`: workspace containing the *multi_uav_sim* ROS package, which contains in turns: 
	* `launch/multi_uav_sim.launch`: launch file to start ROS nodes for exploration and statistics computation;
	* `scripts/mission_test_....py`: ROS node enabling communication of waypoints to Autopilot software through Mavlink
	* `scripts/mavros_test_common_....py`: class useful for handling several services 
	* `scripts/compute_distances.py`: ROS node that computes mutual distances on the basis of poses 
	* `scripts/compute_dist_stats.py`: ROS node computing minimum and average distance during testing 
	* `scripts/missions/plot_waypoints.py`: script to plot waypoints from a flight plan (.plan file) 
	* `scripts/missions/points_to_coords.py`: script to convert waypoints into GPS coordinates an in a flight plan (.plan file) 
	* `scripts/missions/Traj_....csv`: trajectories in [0, 99]x[0, 99]
	* `scripts/missions/Traj_....plan`: flight plan in .json format computed by conversion of .csv trajectory
5. `additional`: folder containing additional launch files and Gazebo world elements for the ROS simulation 
	* `meshes`: folder containing meshes 
	* `models`: folder containing models 
	* `empty_world_physics.world`: empty world 
	* `multi_uav_mavros_sitl_test.launch`: launch file for loading Gazebo world and spawining fleet 
	
## How to run the code 
### 2D environments 
Just run scripts from terminal. An example for training scripts is reported. 
```
python3 main_training.py
```
The working versions for python and python packages are: 
* python 3.9.5
* Tensorflow 2.4.1
* numpy 1.21.2
* scikit-learn 1.0.2
* scikit-image 0.19.2
* scipy 1.8.1
### 3D environments 
For the 3D test environments it is assumed a complete [ROS-Noetic installation](http://wiki.ros.org/noetic/Installation/Ubuntu) with Gazebo and PX4 Autopilot softare (follow [this guide](https://docs.px4.io/v1.12/en/dev_setup/dev_env_linux_ubuntu.html) to install and test installation). 

Build the workspace and source it: 
```
cd catkin_ws
catkin_make
source devel/setup.bash
```


Denoting with *PX4_Path* your PX4 Autopilot installation path:
* Copy */additional/empty_world_physics.world* to *PX4_Path/Tools/sitl_gazebo/worlds/*
* Copy files in */additional/models* to *PX4_Path/Tools/sitl_gazebo/models/*
* Copy */additional/meshes* folder in *PX4_Path/Tools/sitl_gazebo/*
* Copy */additional/multi_uav_mavros_sitl_test.launch* in *PX4_Path/launch/*
* Set some environment variables for simulation 
```
cd PX4_Path
source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/Tools/sitl_gazebo
```
* Initialize the world and spawn UAVs: 
```
roslaunch px4 multi_uav_mavros_sitl_test.launch
```
* Start simulation and flight controllers: 
```
roslaunch multi_uav_sim multi_uav_sim.launch
```

## Notes
The presented code shall be run on a Linux machine, without encoutering relevant issues. 
For the ROS simulations, to add/remove UAVs in the fleet modify `multi_uav_mavros_sitl_test.launch` accordingly, as well as add/remove flight controls in *multi_uav_sim* ROS package for waypoint following. 

## Contact 
Feel free to contact me for any question/suggestion and possible fix suggestions at cosimo.bromo@gmail.com 

## Acknowledgements
The training procedure was developed by modifying [Keras PPO implementation](https://keras.io/examples/rl/ppo_cartpole/). The flight control of the UAVs was implemented by suitably adapting PX4 Autopilot scripts for integration test, available in the [official repository](https://github.com/PX4/PX4-Autopilot). The training scripts here available were run exploiting the computational resources of HPC at Polytechnic of Turin (www.hpc.polito.it) to obtain the trained models. 
