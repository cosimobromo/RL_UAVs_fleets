'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Environment class -------------------

This script is used to define the environment class and methods during training:
- Initialization function:      Allows to read the most important parameters from the parameters dictionary and perform
                                first initializations
- reset:                        Allows to reset the environment for the beginning of a new episode
- get_obst_presence:            Allows to check along the obstacle directions the presence of obstacles and creates the
                                corresponding array
- check_end_episode:            Checks if the action led to episode end or if the maximum number of steps were overcome
- compute_reward:               Allows to compute the current step reward on the basis of the state-action pair
- get_observation:              Allows to perform observation in the neighbourhood of the agent on the basis of the
                                FOV dimension and of its current position. It's just a faster (computationally) way than
                                using lidar_sensor function
- lidar_sensor:                 Simulates a LIDAR sensor for creating a reconstructed map of the neighbourhood of the agent
- set_initial_position:         Initializes randomly the position avoiding placing agent on an obstacle
- get_a_priori_cell:            Computes the position that would be occupied if choosing the action given by the high
                                level (codified in the state)
- step:                         Perform one step moving to the next state, computing reward and checking if action led
                                to episode end

'''

import numpy as np
from useful_functions import *
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import measure

# Class definition
class Env:
    def __init__(self, args):
        # Initialize on the basis of the map dimension (sz1 x sz2)
        self.sz1 = int(args['sz1'])             # x dimension (set by input file)
        self.sz2 = int(args['sz2'])             # y dimension (set by input file)
        self.precision = args['precision']      # Useful for scaling purposes of the map

        # Compute the number of elements in the matrix cost map
        self.n_entries = self.sz1*self.sz2

        # Maximum number of steps per episode
        self.max_length_episode = args['max_episode_length']
        self.Fd = int(args['FOV_dim'])       # Drone FOV dimension (squared area)

        # Define input actions and obstacle directions to seek from
        self.num_actions = int(args['num_actions'])
        self.num_obstacle_dirs = int(args['num_obstacle_dirs'])

        # Safety distance from obstacles for initial positioning of the drone in the map
        self.safety_distance = int(args['safety_distance'])
        self.motion_amount = int(args['motion_amount'])
        self.dt = args['dt']                    # Sampling time

        # Semi-length of the Field of View
        self.Fsd = int((self.Fd - 1) / 2)

        # Other initializations
        self.map = None                     # Contains true map for simulation purposes
        self.nsteps = None                  # Number of steps for each episode

        # Define bounds for observation space and action space
        self.min_position = 0
        self.max_position = 1
        self.low_pos = np.array([self.min_position, self.min_position], dtype = np.float64)
        self.high_pos = np.array([self.max_position, self.max_position], dtype = np.float64)
        self.low_goal = np.array([self.min_position, self.min_position], dtype = np.float64)
        self.high_goal = np.array([self.max_position, self.max_position], dtype = np.float64)

    def reset(self, args):
        '''
        reset function allows to reset the environment at the beginning of each new episode
        '''
        N_training = int(args['N_training_maps'])
        idmap = pick_training_map(N_training)
        map_path = "../Maps/Training_Maps/Training_map_"+str(idmap)
        try:
            self.map = np.genfromtxt(map_path, delimiter=',')  # Load the map
            print("--- Map number " + str(idmap))
        except FileNotFoundError:
            raise FileNotFoundError("Map file in the provided path has not been found")

        self.nsteps = 0                                         # Reset the episode number of steps

        cell = np.squeeze(self.set_initial_position())
        print("Starting cell: ", cell)

        # Convert currently occupied cells in positions, for each UAV it is defined by [x,y] in [0,1]x[0,1]
        pos = compute_UAV_pos(cell, self.sz1, self.sz2)

        # Get observation of the obstacles and the initial action provided
        state = np.zeros((self.num_actions + self.num_obstacle_dirs, ))
        in_action = np.random.choice(self.num_actions)
        state[in_action] = 1

        observed_map, _, _, _, _ = self.get_observation(cell)
        # From the observed map get obstacle observations
        obst_presence = self.get_obst_presence(observed_map)
        state[self.num_actions:] = obst_presence

        return state, pos, cell

    def get_obst_presence(self, observed_map):
        '''
        Function to create the obst_presence list, containing 0 for directions without obstacles (in the detection
        range) and 1 in the directions with an obstacle (in the detection range).
        '''
        obst_presence = np.zeros((self.num_obstacle_dirs,))

        # Check if there are obstacles in the current position (action 0)
        if observed_map[self.Fsd, self.Fsd] == 1:
            obst_presence[0] = 1

        # Check if there are obstacles in the up direction
        for idx in range(1, self.motion_amount + 1):
            if observed_map[self.Fsd - idx, self.Fsd] == 1:
                obst_presence[1] = 1
                break

        # Check if there are obstacles in the down direction
        for idx in range(1, self.motion_amount + 1):
            if observed_map[self.Fsd + idx, self.Fsd] == 1:
                obst_presence[2] = 1
                break

        # Check if there are obstacles in the left direction
        for idx in range(1, self.motion_amount + 1):
            if observed_map[self.Fsd, self.Fsd - idx] == 1:
                obst_presence[3] = 1
                break

        # Check if there are obstacles in the right direction
        for idx in range(1, self.motion_amount + 1):
            if observed_map[self.Fsd, self.Fsd + idx] == 1:
                obst_presence[4] = 1
                break

        # Check if there are obstacles in the up left direction
        for idx in range(1, self.motion_amount + 1):
            if observed_map[self.Fsd - idx, self.Fsd - idx] == 1:
                obst_presence[5] = 1
                break

        # Check if there are obstacles in the up right direction
        for idx in range(1, self.motion_amount + 1):
            if observed_map[self.Fsd - idx, self.Fsd + idx] == 1:
                obst_presence[6] = 1
                break

        # Check if there are obstacles in the down left direction
        for idx in range(1, self.motion_amount + 1):
            if observed_map[self.Fsd + idx, self.Fsd - idx] == 1:
                obst_presence[7] = 1
                break

        # Check if there are obstacles in the down right direction
        for idx in range(1, self.motion_amount + 1):
            if observed_map[self.Fsd + idx, self.Fsd + idx] == 1:
                obst_presence[8] = 1
                break
        return obst_presence

    def check_end_episode(self, new_cell):
        '''
        The episode is considered to be over in the following cases:
        1 - More than a certain percentage of coverage has been attained
        2 - Maximum number of steps has been reached
        '''
        done = False
        info = 0
        if self.map[new_cell[0], new_cell[1]]:
            print("Collision with obstacle")
            info = 1
            done = True
        if self.nsteps >= self.max_length_episode:
            print("Maximum number of steps (" + str(int(self.max_length_episode)) + ") reached")
            info = 2
            done = True

        return done, info

    def compute_reward(self, new_cell, priori_cell, state, action):
        '''
        Compute reward on the basis of new cell, obstacle presence, a priori cell (see get_a_priori_cell method for
        further information)
        '''
        initial_input_action = np.where(state == 1)[0][0]
        obst_presence = state[self.num_actions:]

        if initial_input_action == action.numpy()[0]:
            if obst_presence[action.numpy()[0]] == 0:
                reward = +1
            else:
                reward = -1
        else:  # Different actions
            if obst_presence[initial_input_action] == 1:
                if obst_presence[action.numpy()[0]] == 0:
                    reward = +1 - np.linalg.norm(np.array(new_cell) - np.array(priori_cell))/(np.sqrt(2)*2*self.motion_amount+1)
                else:
                    reward = -1
            else:
                if obst_presence[action.numpy()[0]] == 0:
                    reward = -0.5
                else:
                    reward = -1

        return reward

    def get_observation(self, current_cell):
        '''
        This function allows to simulate the observation by the UAV(s) on the basis of the currently occupied cell(s).

        Input           --->    Current occupied cell in [0, sz1-1] x [0, sz2-1]
        Output          --->
                                - Fd x Fd map of the current observed FOV (properly adjusted in order to have always the same dimension)
                                - min and max indices (related to the overall map) for updating purposes
        '''

        # Once cells are computed, compute the interval of interest for both sizes
        min_x = max(0, current_cell[0] - self.Fsd)
        max_x = min(self.sz1 - 1, current_cell[0] + self.Fsd)
        min_y = max(0, current_cell[1] - self.Fsd)
        max_y = min(self.sz2 - 1, current_cell[1] + self.Fsd)

        # It is needed to avoid that because of out of bounds position, some unwanted observation is performed:
        if min_x < 0:
            min_x = 0
        if max_x > self.sz1-1:
            max_x = self.sz1-1
        if min_y < 0:
            min_y = 0
        if max_y > self.sz2-1:
            max_y = self.sz2-1

        # If the action is too big, it may happen that min > max because of FOV_sd

        if min_x == 0 and max_x < min_x:
            max_x = 0
        if min_y == 0 and max_y < min_y:
            max_y = 0
        if max_x == self.sz1-1 and min_x > max_x:
            min_x = self.sz1-1
        if max_y == self.sz2-1 and min_y > max_y:
            min_y = self.sz2-1

        observed_map = self.map[min_x:max_x+1, min_y:max_y+1].copy()
        size_observation = observed_map.shape

        # Adding padding to the observed matrix to have always the same size (useful in the case of my own Path Planning Agent)
        if min_x == 0:
            els = int(self.Fd-size_observation[0])
            add_mat = np.ones((els, size_observation[1]))
            observed_map = np.concatenate((add_mat, observed_map), axis = 0)
        size_observation = observed_map.shape
        if max_x == self.sz1-1:
            els = int(self.Fd-size_observation[0])
            add_mat = np.ones((els, size_observation[1]))
            observed_map = np.concatenate((observed_map, add_mat), axis = 0)
        size_observation = observed_map.shape
        if min_y == 0:
            els = int(self.Fd-size_observation[1])
            add_mat = np.ones((size_observation[0], els))
            observed_map = np.concatenate((add_mat, observed_map), axis = 1)
        size_observation = observed_map.shape
        if max_y == self.sz2-1:
            els = int(self.Fd-size_observation[1])
            add_mat = np.ones((size_observation[0], els))
            observed_map = np.concatenate((observed_map, add_mat), axis = 1)

        return observed_map, min_x, max_x, min_y, max_y

    def set_initial_position(self):
        '''
        Randomly initializes the position of the UAV in the grid
        Output ---> Initial cell representing UAV position
        '''
        flag = False
        while flag is False:
            # Define an initial position by randomly picking a value in [0, sz1-1] and [0, sz2-1]
            idx = np.random.choice(self.sz1)
            idy = np.random.choice(self.sz2)

            # Check that no obstacle is present in a square of distance safety_distance from the drone
            min_x = max(0, idx - self.safety_distance)
            max_x = min(self.sz1 - 1, idx + self.safety_distance)
            min_y = max(0, idy - self.safety_distance)
            max_y = min(self.sz2 - 1, idy + self.safety_distance)
            surrounding_matrix = self.map[min_x:max_x + 1, min_y:max_y + 1]
            if np.any(surrounding_matrix) == False:
                flag = True

        cells = cast_to_integer(np.array([idx, idy], dtype=np.float32))
        return cells

    def get_a_priori_cell(self, state, cell):
        '''
        This function is used in order to compute the cell that would be occupied if moving according to the action
        indicated by the state (used for reward computation purposes)
        '''
        motion_amount = self.motion_amount
        action = np.where(state == 1)[0][0]
        current_row = cell[0]
        current_col = cell[1]

        if action == 1:  # Move up
            current_row = max(current_row - motion_amount, 0)

        if action == 2:  # Move down
            current_row = min(current_row + motion_amount, self.sz1 - 1)

        if action == 3:  # Move left
            current_col = max(current_col - motion_amount, 0)

        if action == 4:  # Move right
            current_col = min(current_col + motion_amount, self.sz2 - 1)

        if action == 5:  # Move up left
            current_row = max(current_row - motion_amount, 0)
            current_col = max(current_col - motion_amount, 0)

        if action == 6:  # Move up right
            current_row = max(current_row - motion_amount, 0)
            current_col = min(current_col + motion_amount, self.sz2 - 1)

        if action == 7:  # Move down left
            current_row = min(current_row + motion_amount, self.sz1 - 1)
            current_col = max(current_col - motion_amount, 0)

        if action == 8:  # Move down right
            current_row = min(current_row + motion_amount, self.sz1 - 1)
            current_col = min(current_col + motion_amount, self.sz2 - 1)

        priori_cell = [current_row, current_col]

        return priori_cell

    def step(self, state, cell, action):
        '''
            On the basis of the high level action (in the overall implementation it is provided by the coverage agent)
            and on the obstacle detections, selects an action to take:
            - 0: do nothing
            - 1: move up
            - 2: move down
            - 3: move left
            - 4: move right
            - 5: move up left
            - 6: move up right
            - 7: move down left
            - 8: move down right

            Any motion leads to a motion of motion_amount cells.

            For each agent:
            - take computed action
            - compute reward
            - randomly select the next high level action and compute new state
        '''
        self.nsteps += 1
        motion_amount = self.motion_amount

        priori_cell = self.get_a_priori_cell(state, cell)

        current_row = cell[0]
        current_col = cell[1]

        if action == 1: # Move up
            current_row = max(current_row - motion_amount, 0)

        if action == 2: # Move down
            current_row = min(current_row + motion_amount, self.sz1-1)

        if action == 3: # Move left
            current_col = max(current_col - motion_amount, 0)

        if action == 4: # Move right
            current_col = min(current_col + motion_amount, self.sz2-1)

        if action == 5: # Move up left
            current_row = max(current_row - motion_amount, 0)
            current_col = max(current_col - motion_amount, 0)

        if action == 6: # Move up right
            current_row = max(current_row - motion_amount, 0)
            current_col = min(current_col + motion_amount, self.sz2-1)

        if action == 7: # Move down left
            current_row = min(current_row + motion_amount, self.sz1-1)
            current_col = max(current_col - motion_amount, 0)

        if action == 8: # Move down right
            current_row = min(current_row + motion_amount, self.sz1-1)
            current_col = min(current_col + motion_amount, self.sz2-1)

        new_cell = [current_row, current_col]

        # Compute the reward
        reward = self.compute_reward(new_cell, priori_cell, state, action)

        # Check if episode shall end and why
        done, info = self.check_end_episode(new_cell)

        new_action_high_level = np.random.choice(self.num_actions)
        new_state = np.zeros((self.num_actions+self.num_obstacle_dirs,))
        new_state[new_action_high_level] = 1
        new_observed_map, _, _, _, _ = self.get_observation(new_cell)
        # From the observed map get obstacle observations
        obst_presence = self.get_obst_presence(new_observed_map)
        new_state[self.num_actions:] = obst_presence

        return new_state, new_cell, reward, done, info

    def lidar_sensor(self, current_cell):
        '''
        LIDAR sensor simulation, computationally heavier than get_observation, but perfectly working

        Note: In training implementation is not used in order to reduce the computational burden
        '''
        observed_map, min_x, max_x, min_y, max_y = self.get_observation(current_cell)

        # Compute the maximum radius for obstacle detection (simulating the LIDAR sensor)
        # This will turn into an outer bounding matrix from which to extract the true observation
        diag_length = np.sqrt(2) * self.Fd * self.precision  # Gives the maximum diagonal of detection

        # Compute the center
        temporary_obs = np.zeros_like(observed_map)  # To be passed to the agent
        obs_for_cov = np.zeros_like(observed_map)  # To be used for coverage update
        center = self.Fsd
        dim = self.Fd * self.precision  # In cm
        radii = np.arange(0, diag_length / 2, self.precision / 100)
        angles = np.arange(0, 2 * np.pi - np.pi / 100, np.pi / 100)
        for angle in angles:
            flag = 0
            for radius in radii:
                if flag == 0:
                    pos_mf = np.array(
                        [radius * np.cos(angle), radius * np.sin(angle)])  # In the meausurement reference frame
                    pos_rf = np.array(
                        [dim / 2 - pos_mf[1], dim / 2 + pos_mf[0]])  # Current position in the reference frame
                    pos_rescaled = pos_rf / np.array([dim, dim])
                    if any(pos_rescaled < 0) or any(pos_rescaled > 1):
                        pass
                    else:
                        # Compute corresponding cell
                        cell = compute_UAV_cell(pos_rescaled, self.Fd, self.Fd)
                        if observed_map[cell[0]][cell[1]] == 1:
                            temporary_obs[cell[0]][cell[1]] = 1
                            obs_for_cov[cell[0]][cell[1]] = 1
                            flag = 1  # To put to 1 all the consequent elements in temporary obs
                else:
                    pos_mf = np.array(
                        [radius * np.cos(angle), radius * np.sin(angle)])  # In the meausurement reference frame
                    pos_rf = np.array(
                        [dim / 2 - pos_mf[1], dim / 2 + pos_mf[0]])  # Current position in the reference frame
                    pos_rescaled = pos_rf / np.array([dim, dim])
                    if any(pos_rescaled < 0) or any(pos_rescaled > 1):
                        pass
                    else:
                        # Compute corresponding cell
                        cell = compute_UAV_cell(pos_rescaled, self.Fd, self.Fd)
                        temporary_obs[cell[0]][cell[1]] = 1

        return temporary_obs, obs_for_cov, observed_map, min_x, max_x, min_y, max_y
