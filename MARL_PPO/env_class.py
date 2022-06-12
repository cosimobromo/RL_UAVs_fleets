'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Environment class -------------------

This script is used to define the environment class and methods during training:
- Initialization function:      Allows to read the most important parameters from the parameters dictionary and perform
                                first initializations
- add_borders:                  Adds 1s to the coverage and reconstructed map on the borders (surely present an obstacle
                                there)
- reset:                        Allows to reset the environment for the beginning of a new episode during training
- reset_test:                   Allows to reset the environment for the beginning of a new episode during testing
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
- step_test:                    Perform one step moving to the next state in test conditions (calling the OA agent)
'''

import numpy as np
from useful_functions import *
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import measure

# Class definition
class Env:
    def __init__(self, args):
        '''
        Environment class (Env) defines the environment and adapts the interaction of the agent(s) with it.

        At initialization of the environment, set some important characteristics:
        - N                                 Number of agents (completely identical each other) to be properly trained
        - sz1, sz2                          Sizes of the environment to be simulated (in terms of grid dimensions):
                                            - sz1: x dimension
                                            - sz2: y dimension
        - precision                         In [m], defines the scaling to real-word simulation environments
        - n_entries                         Defines the total number of grids in the environment grid (sz1 x sz2)
        - N_actions                         Defines the number of possible actions
        - max_length                        Maximum number of steps per epoch
        - max_episode_length                Maximum number of steps per episode
        - Fd                                Drone FOV (Field of View) dimension
        - Fsd                               Semi-length of the FOV
        - safety_distance                   Safety distance for initial positioning of drones inside the map from obstacles
        - mutual_initial_safety_distance    Mutual initial safety distance for initial positioning of drones inside the map
        - motion amount                     Number of cells to move
        - obstacle range detection          Detection in terms of cells
        - dt                                Sampling time (default = 1)
        - coverage_threshold                Percentage of the map to consider it fully covered
        - reciprocal_distance_threshold     Reciprocal distance (with rescaled positions) to consider collision occurrance among UAVs
        - map                               True map of the current simulation environment
        - covmap                            Coverage map (binary map) for taking into account covered area
        - recmap                            Reconstructed map, on the basis of UAV's observations
        - obstmap                           Map containing discovered obstacles
        - nsteps                            Counts the number of steps per episode
        - UAV_collisions                    Counts the number of collisions among UAVs per episode
        - obst_collisions                   Counts the number of collisions with obstacles per episode
        - coverage_per_step_list            Contains at each step, the overall coverage of the map, during the episode
        - num_obstacle_dirs                 Number of directions in which sensors can detect obstacles
        - min_position                      Defines the minimum bound for positions
        - max_position                      Defines the maximum bound for positions
        - low_pos                           Defines array of lower bounds for positions
        - high_pos                          Defines array of upper bounds for positions
        - low_goal                          Defines array of lower bounds for goals
        - high_goal                         Defines array of upper bounds for goals
        '''

        self.N = int(args['N_agents'])
        self.sz1 = int(args['sz1'])
        self.sz2 = int(args['sz2'])
        self.precision = args['precision']
        self.n_entries = self.sz1*self.sz2
        self.N_actions = int(args['num_actions'])
        self.max_length = args['max_length']
        self.max_episode_length = args['max_episode_length']
        self.Fd = int(args['FOV_dim'])
        self.Fsd = int((self.Fd - 1) / 2)
        self.safety_distance = int(args['safety_distance'])
        self.mutual_initial_safety_distance = int(args['mutual_initial_safety_distance'])
        self.motion_amount = int(args['motion_amount'])
        self.obstacle_range_detection = int(args['obstacle_range_detection'])
        self.dt = args['dt']
        self.coverage_threshold = args['coverage_threshold']
        self.reciprocal_distance_threshold = args['reciprocal_distance_threshold']
        self.map = None
        self.covmap = None
        self.recmap = None
        self.obstmap = None
        self.nsteps = None
        self.UAV_collisions = None
        self.obst_collisions = None
        self.coverage_per_step_list = None
        self.num_obstacle_dirs = int(args['num_obstacle_dirs'])
        self.min_position = 0
        self.max_position = 1
        self.low_pos = np.array([self.min_position, self.min_position], dtype = np.float64)
        self.high_pos = np.array([self.max_position, self.max_position], dtype = np.float64)
        self.low_goal = np.array([self.min_position, self.min_position], dtype = np.float64)
        self.high_goal = np.array([self.max_position, self.max_position], dtype = np.float64)

    def add_borders(self):
        self.covmap[0,:] = 1
        self.covmap[-1,:] = 1
        self.covmap[:,0] = 1
        self.covmap[:,-1] = 1
        self.recmap[0, :] = 1
        self.recmap[-1, :] = 1
        self.recmap[:, 0] = 1
        self.recmap[:, -1] = 1

    def reset(self, args):
        '''
        At the beginning of each new episode:
        - Pick a map as simulation map and store it in Env.map array for simulation purposes
        - Initialize steps counter
        - Initialize to zero arrays the coverage and reconstructed maps
        - Initialize the collisions counter
        - Initialize positions of the N UAVs and save their corresponding cell occupations
        - Update reconstructed and coverage maps on the basis of initial positions
        - Compute current positions in [0,1]x[0,1]
        - Store coverage map in a temporary array 'cov_map'
        - For each UAV, compute the state input, which is a 2 layer image:
            - First layer is the position map, taking into account all UAV's positions (it is different among each UAV)
            - Second layer contains the coverage map
        - Initialize properly the structure to consider all the trajectories initialization
        '''
        N_training = int(args['N_training_maps'])
        idmap = pick_training_map(N_training)
        map_path = "../Maps/Training_Maps/Training_map_"+str(idmap)
        try:
            self.map = np.genfromtxt(map_path, delimiter=',')  # Load the map
        except FileNotFoundError:
            raise FileNotFoundError("Map file in the provided path has not been found")
        self.nsteps = 0
        self.covmap = np.zeros_like(self.map)
        self.recmap = np.zeros_like(self.map)
        self.obstmap = np.zeros_like(self.map)
        self.UAV_collisions = 0
        self.obst_collisions = 0
        #self.add_borders()
        cells = np.squeeze(self.initialize_positions())
        for cell in cells:
            self.observe_and_update(cell)
        positions = compute_UAV_pos(cells, self.sz1, self.sz2)
        cov_map = self.covmap.copy()
        rec_map = self.recmap.copy()
        obst_map = self.obstmap.copy()
        states = []
        for id_agent in range(self.N):
            pos_map = create_pos_map(cells, id_agent, self.N, kernel_size = self.Fd)
            state = np.zeros((self.sz1, self.sz2, 2))
            state[:, :, 0] = pos_map.copy()
            state[:, :, 1] = cov_map.copy()
            states.append(state)

        trajectories = []
        for id_ag in range(self.N):
            trajectories.append([])
            trajectories[id_ag].append(cells[id_ag])

        statistics = []
        self.coverage_per_step_list = []

        return states, positions, cells, trajectories, statistics

    def reset_test(self, args, random_choice = False, path = None, mut_dist = 15):
        '''
        At the beginning of each new episode:
        - Pick a map as simulation map and store it in Env.map array for simulation purposes
        - Initialize steps counter
        - Initialize to zero arrays the coverage and reconstructed maps
        - Initialize the collisions counter
        - Initialize positions of the N UAVs and save their corresponding cell occupations
        - Update reconstructed and coverage maps on the basis of initial positions
        - Compute current positions in [0,1]x[0,1]
        - Store coverage map in a temporary array 'cov_map'
        - For each UAV, compute the state input, which is a 2 layer image:
            - First layer is the position map, taking into account all UAV's positions (it is different among each UAV)
            - Second layer contains the coverage map
        - Initialize properly the structure to consider all the trajectories initialization

        random_choice parameter allows to select whether to choose randomly the map among the test set or to provide a
        specific path (including map name) to test
        '''

        if random_choice:
            N_testing = int(args['N_testing_maps'])
            idmap = pick_training_map(N_testing)
            map_path = "../Maps/Testing_Maps/Testing_map_"+str(idmap)
        else:
            if path is None:
                print("Path not provided!")
            else:
                map_path = path

        try:
            self.map = np.genfromtxt(map_path, delimiter=',')  # Load the map
        except FileNotFoundError:
            raise FileNotFoundError("Map not found in " + map_path)
        self.nsteps = 0
        self.covmap = np.zeros_like(self.map)
        self.recmap = np.zeros_like(self.map)
        self.obstmap = np.zeros_like(self.map)
        self.UAV_collisions = 0
        self.obst_collisions = 0
        self.add_borders()
        cells = np.squeeze(self.initialize_positions(random = False, mutual_distance = mut_dist))
        for cell in cells:
            self.observe_and_update(cell)
        positions = compute_UAV_pos(cells, self.sz1, self.sz2)
        cov_map = self.covmap.copy()
        rec_map = self.recmap.copy()
        obst_map = self.obstmap.copy()
        states = []
        for id_agent in range(self.N):
            pos_map = create_pos_map(cells, id_agent, self.N, kernel_size = self.Fd)
            state = np.zeros((self.sz1, self.sz2, 2))
            state[:, :, 0] = pos_map.copy()
            state[:, :, 1] = cov_map.copy()
            states.append(state)

        trajectories = []
        for id_ag in range(self.N):
            trajectories.append([])
            trajectories[id_ag].append(cells[id_ag])

        statistics = []
        self.coverage_per_step_list = []

        return states, positions, cells, trajectories, statistics


    def check_end_episode(self, coverage_perc):
        '''
        The episode is considered to be over in the following cases:
        1 - More than a certain percentage of coverage has been attained
        2 - Maximum number of steps has been reached
        '''
        done = False
        info = 0
        if coverage_perc >= self.coverage_threshold:
            print("More than " + str(self.coverage_threshold * 100) + "% of the area has been covered")
            # Fix the coverage for the following steps equal to the final one
            while len(self.coverage_per_step_list) < self.max_episode_length:
                self.coverage_per_step_list.append(coverage_perc)
            info = 1
            done = True
        if self.nsteps >= self.max_episode_length:
            print("Maximum number of steps (" + str(int(self.max_episode_length)) + ") reached")
            info = 2
            done = True

        return done, info

    def compute_reward(self, individual_coverage_increases, cov_perc, positions):
        '''
        Rewards are the sum of 2 contributions:
        - coverage increase * K_coverage if coverage increase > 0 else -0.1 (if no coverage increase)
        - -0.5 if two agents are too near (under a given specified threshold)
        - + 10 if current action allows to cover the entire map
        '''

        rewards = [0.01*coverage_increase if coverage_increase > 0 else -0.1 for coverage_increase in individual_coverage_increases]
        completed_coverage_reward = 10
        collision_risk_reward = -0.5

        if cov_perc >= self.coverage_threshold:
            for id_agent in range(self.N):
                ind_comp_cov = completed_coverage_reward if individual_coverage_increases[id_agent] > 0 else 0
                rewards[id_agent] += ind_comp_cov

        distances, bad_agents_ids = compute_mutual_distances(positions, self.reciprocal_distance_threshold)

        # Negative mutual collision reward
        for bad_agent in bad_agents_ids:
            rewards[bad_agent] += collision_risk_reward
            self.UAV_collisions += 0.5                          # Add 0.5 since each collision is considered twice

        # Compute statistics
        std_pos = 0.5 * (np.std(positions[:, 0]) + np.std(positions[:, 1]))
        std_distances = np.std(distances)
        avg_distances = np.mean(distances)
        min_distance = np.min(distances)

        stats = [std_pos, std_distances, avg_distances, min_distance]
        return rewards, stats

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

    def get_observation_obstacle_det(self, current_cell, current_cells, current_id):
        '''
        This function allows to simulate the observation by the UAV(s) on the basis of the currently occupied cell(s).

        Input           --->    Current occupied cell in [0, sz1-1] x [0, sz2-1]
        Output          --->
                                - Fd x Fd map of the current observed FOV (properly adjusted in order to have always the same dimension)
                                - min and max indices (related to the overall map) for updating purposes
        '''
        obst_det_Fd = int(self.obstacle_range_detection*2+1)
        # Once cells are computed, compute the interval of interest for both sizes
        min_x = max(0, current_cell[0] - self.obstacle_range_detection)
        max_x = min(self.sz1 - 1, current_cell[0] + self.obstacle_range_detection)
        min_y = max(0, current_cell[1] - self.obstacle_range_detection)
        max_y = min(self.sz2 - 1, current_cell[1] + self.obstacle_range_detection)

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

        # To consider other agents as obstacles
        pos_map_agents = np.zeros_like(self.map)
        for id_agent in range(self.N):
            if id_agent != current_id:
                pos_map_agents[max(current_cells[id_agent, 0]-1, 0):min(current_cells[id_agent, 0]+1, self.sz1-1)+1, max(current_cells[id_agent, 1]-1, 0):min(current_cells[id_agent, 1]+1, self.sz2-1)+1] = 1

        observed_map = self.map[min_x:max_x+1, min_y:max_y+1].copy() + pos_map_agents[min_x:max_x+1, min_y:max_y+1].copy()
        size_observation = observed_map.shape

        # Adding padding to the observed matrix to have always the same size (useful in the case of my own Path Planning Agent)
        if min_x == 0:
            els = int(obst_det_Fd-size_observation[0])
            add_mat = np.ones((els, size_observation[1]))
            observed_map = np.concatenate((add_mat, observed_map), axis = 0)
        size_observation = observed_map.shape
        if max_x == self.sz1-1:
            els = int(obst_det_Fd-size_observation[0])
            add_mat = np.ones((els, size_observation[1]))
            observed_map = np.concatenate((observed_map, add_mat), axis = 0)
        size_observation = observed_map.shape
        if min_y == 0:
            els = int(obst_det_Fd-size_observation[1])
            add_mat = np.ones((size_observation[0], els))
            observed_map = np.concatenate((add_mat, observed_map), axis = 1)
        size_observation = observed_map.shape
        if max_y == self.sz2-1:
            els = int(obst_det_Fd-size_observation[1])
            add_mat = np.ones((size_observation[0], els))
            observed_map = np.concatenate((observed_map, add_mat), axis = 1)

        return observed_map, min_x, max_x, min_y, max_y

    def update_maps(self, min_x, max_x, min_y, max_y):
        '''
        Allows to update the current maps (coverage map and reconstructed map) on the basis of the new observation
        '''

        # Reconstructed map update
        for i in range(min_x, max_x+1):
            for j in range(min_y, max_y+1):
                if self.map[i,j] == 1:
                    self.recmap[i,j] = 1
                else:
                    self.recmap[i,j] = 0.5

        # Perform obstacle shape prediction (by now on rectangular shaped obstacles only) on the reconstructed map
        proc_rec_map = self.recmap.copy()
        proc_rec_map = np.delete(proc_rec_map, (0, -1), axis = 0)   # Remove first and last row (surely an obstacle)
        proc_rec_map = np.delete(proc_rec_map, (0, -1), axis = 1)   # Remove first and last col (surely an obstacle)

        # Find contours of the different obstacles
        contours = measure.find_contours(proc_rec_map, 0.99)
        # Fill the spaces properly
        for contour in contours:    # For any obstacle predicted and found
            mx = int(np.round(min(contour[:, 0])))
            my = int(np.round(min(contour[:, 1])))
            Mx = int(np.round(max(contour[:, 0])))
            My = int(np.round(max(contour[:, 1])))
            proc_rec_map[mx:Mx+1, my:My+1] = 1

        self.recmap[1:-1, 1:-1] = proc_rec_map.copy() # Update properly the reconstructed map using the predicted obstacles

        # Now update the coverage map
        self.covmap = np.clip(self.recmap*2, np.zeros_like(self.recmap), np.ones_like(self.recmap))
        # Now update the obstacle map
        self.obstmap = np.array([self.recmap[i,j] if self.recmap[i,j] == 1 else 0 for i in range(self.sz1) for j in range(self.sz2)]).reshape((self.sz1, self.sz2))

    def observe_and_update(self, current_cell):
        '''
        This function has the aim, for each UAV, to call the "get_observation" function and update consistently the coverage and reconstructed maps
        '''
        observed_map, min_x, max_x, min_y, max_y = self.get_observation(current_cell)
        self.update_maps(min_x, max_x, min_y, max_y)

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

    def initialize_positions(self, random = True, mutual_distance = 10):
        '''
        This function initializes positions of the different UAVs either randomly, exploiting set_initial_position()
        function and placing UAVs at a minimum distance of "mutual_initial_safety_distance" cells each other, or in one
        corner with a triangular distribution and a mutual distance of a modifyable number of cells
        '''
        if random:  # Random = True
            cell_matrix = np.zeros((self.N, 2))
            completed = False
            id_ag = 0
            while not completed:
                selected_cell = self.set_initial_position()
                # Check that the currently selected position is far from the other drones at least 10 cells
                flag = False
                for j in range(id_ag):
                    if max(np.abs(cell_matrix[j] - selected_cell)) <= self.mutual_initial_safety_distance:
                        flag = True     # Not possible to place the obstacle here!
                if flag:
                    pass  # Perform assignment again
                else:
                    cell_matrix[id_ag,:] = selected_cell.copy()
                    id_ag += 1

                if id_ag == self.N:    # Position has been properly assigned for all agents
                    completed = True
        else:       # Random = False
            cell_matrix = np.zeros((self.N, 2))
            assigned = 0
            cell = [self.Fsd + 2, self.Fsd + 2]
            while assigned < self.N:
                if self.map[cell[0], cell[1]] == 0:     # No obstacle is present
                    cell_matrix[assigned,:] = cell
                    assigned += 1
                if cell[0] == self.Fsd + 2:
                    cell[0] = cell[1] + mutual_distance
                    cell[1] = self.Fsd + 2
                else:
                    cell[0] = cell[0] - mutual_distance
                    cell[1] = cell[1] + mutual_distance
        return cast_to_integer(cell_matrix)

    def compute_mutual_distances(self, positions):
        '''
        This function allows to compute mutual distances (in Euclidean norm) among UAVs for statistical purposes
        '''
        distances = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                distances.append(np.linalg.norm(positions[i,:]-positions[j,:]))

        return distances

    def step(self, cells, agents_actions):
        '''
        For each agent, on the basis of the current position, apply the selected action
            Consider the current values of action mapping to the actions to take:
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
            - compute individual coverage increase
            - initialize coverage map to previous conditions

            Once computed individual coverage increases:
            - update the overall coverage map
            - compute rewards for each UAV
            - compute new states for next step

        '''
        self.nsteps += 1
        initial_cov_map = self.covmap.copy()
        initial_rec_map = self.recmap.copy()
        individual_coverage_increases = []
        motion_amount = self.motion_amount
        new_cells = []

        for id_agent in range(self.N):
            current_row = cells[id_agent][0]
            current_col = cells[id_agent][1]
            action = agents_actions[id_agent]

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

            self.observe_and_update([current_row, current_col])
            cov_map = self.covmap.copy()
            individual_coverage_increases.append(increase_coverage(initial_cov_map, cov_map))
            new_cells.append(np.array([current_row, current_col]))
            self.covmap = initial_cov_map.copy()
            self.recmap = initial_rec_map.copy()
        for new_cell in new_cells:
            self.observe_and_update(new_cell)
        covperc = coverage_percentage(self.covmap)
        self.coverage_per_step_list.append(covperc)
        positions = compute_UAV_pos(new_cells, self.sz1, self.sz2)
        rewards, stats= self.compute_reward(individual_coverage_increases, covperc, positions)
        done, info = self.check_end_episode(covperc)

        # Create new states
        cov_map = self.covmap.copy()
        rec_map = self.recmap.copy()
        obst_map = self.obstmap.copy()
        new_states = []
        new_cells = np.array(new_cells)
        for id_agent in range(self.N):
            pos_map = create_pos_map(new_cells, id_agent, self.N, kernel_size = self.Fd)
            new_state = np.zeros((self.sz1, self.sz2, 2))
            new_state[:, :, 0] = pos_map.copy()
            new_state[:, :, 1] = cov_map.copy()
            new_states.append(new_state)

        return new_states, new_cells, rewards, done, info, stats

    def get_obst_presence(self, observed_map):
        '''
        Function to create the obst_presence list, containing 0 for directions without obstacles (in the detection
        range) and 1 in the directions with an obstacle (in the detection range).
        '''
        obst_presence = np.zeros((self.num_obstacle_dirs,))
        # Check if there are obstacles in the current position (action 0)
        if observed_map[self.obstacle_range_detection, self.obstacle_range_detection] == 1:
            obst_presence[0] = 1

        # Check if there are obstacles in the up direction
        for idx in range(1, self.obstacle_range_detection + 1):
            if observed_map[self.obstacle_range_detection - idx, self.obstacle_range_detection] == 1:
                obst_presence[1] = 1
                break

        # Check if there are obstacles in the down direction
        for idx in range(1, self.obstacle_range_detection + 1):
            if observed_map[self.obstacle_range_detection + idx, self.obstacle_range_detection] == 1:
                obst_presence[2] = 1
                break

        # Check if there are obstacles in the left direction
        for idx in range(1, self.obstacle_range_detection + 1):
            if observed_map[self.obstacle_range_detection,  self.obstacle_range_detection - idx] == 1:
                obst_presence[3] = 1
                break

        # Check if there are obstacles in the right direction
        for idx in range(1, self.obstacle_range_detection + 1):
            if observed_map[self.obstacle_range_detection, self.obstacle_range_detection + idx] == 1:
                obst_presence[4] = 1
                break

        # Check if there are obstacles in the up left direction
        for idx in range(1, self.obstacle_range_detection + 1):
            if observed_map[self.obstacle_range_detection - idx, self.obstacle_range_detection - idx] == 1:
                obst_presence[5] = 1
                break

        # Check if there are obstacles in the up right direction
        for idx in range(1, self.obstacle_range_detection + 1):
            if observed_map[self.obstacle_range_detection - idx, self.obstacle_range_detection + idx] == 1:
                obst_presence[6] = 1
                break

        # Check if there are obstacles in the down left direction
        for idx in range(1, self.obstacle_range_detection + 1):
            if observed_map[self.obstacle_range_detection + idx, self.obstacle_range_detection - idx] == 1:
                obst_presence[7] = 1
                break

        # Check if there are obstacles in the down right direction
        for idx in range(1, self.obstacle_range_detection + 1):
            if observed_map[self.obstacle_range_detection + idx, self.obstacle_range_detection + idx] == 1:
                obst_presence[8] = 1
                break
        return obst_presence

    def step_test(self, cells, agents_actions, oa_model, num_inputs_oa):
        '''
        For each agent, on the basis of the current position, apply the selected action
            Consider the current values of action mapping to the actions to take:
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
            - compute individual coverage increase
            - initialize coverage map to previous conditions

            Once computed individual coverage increases:
            - update the overall coverage map
            - compute rewards for each UAV
            - compute new states for next step
        '''
        self.nsteps += 1
        initial_cov_map = self.covmap.copy()
        initial_rec_map = self.recmap.copy()
        individual_coverage_increases = []
        motion_amount = self.motion_amount

        new_cells = []

        for id_agent in range(self.N):
            observed_map, _, _, _, _ = self.get_observation_obstacle_det(cells[id_agent], cells, id_agent)
            obst_pres = self.get_obst_presence(observed_map)
            oa_state = np.zeros((num_inputs_oa,))
            cov_action = agents_actions[id_agent]
            oa_state[cov_action.numpy()] = 1
            oa_state[int(num_inputs_oa/2):] = obst_pres

            # Using agent from PPO
            logits = oa_model(tf.reshape(tf.convert_to_tensor(oa_state), (1,-1)))
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=1).numpy()
            #action = tf.math.argmax(logits, 1).numpy()

            current_row = cells[id_agent][0]
            current_col = cells[id_agent][1]

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

            self.observe_and_update([current_row, current_col])
            cov_map = self.covmap.copy()
            individual_coverage_increases.append(increase_coverage(initial_cov_map, cov_map))
            new_cells.append(np.array([current_row, current_col]))
            self.covmap = initial_cov_map.copy()
            self.recmap = initial_rec_map.copy()
        for new_cell in new_cells:
            self.observe_and_update(new_cell)

        covperc = coverage_percentage(self.covmap)
        self.coverage_per_step_list.append(covperc)
        positions = compute_UAV_pos(new_cells, self.sz1, self.sz2)
        rewards, stats= self.compute_reward(individual_coverage_increases, covperc, positions)
        done, info = self.check_end_episode(covperc)

        # Create new states
        cov_map = self.covmap.copy()
        rec_map = self.recmap.copy()
        obst_map = self.obstmap.copy()
        new_states = []
        new_cells = np.array(new_cells)
        for id_agent in range(self.N):
            pos_map = create_pos_map(new_cells, id_agent, self.N, kernel_size = self.Fd)
            new_state = np.zeros((self.sz1, self.sz2, 2))
            new_state[:, :, 0] = pos_map.copy()
            new_state[:, :, 1] = cov_map.copy()
            new_states.append(new_state)

        return new_states, new_cells, rewards, done, info, individual_coverage_increases, stats

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
