'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Creation of maps for training and testing --------------------------

This project creates 3000 maps (90 % for training and 10% for testing) with an obstacle occupancy ranging from 0% to 20%

'''

import numpy as np
import os
import shutil
from maps_creation import Map

# Create new directories for maps savings
try:
    os.mkdir("Training_Maps")
except FileExistsError:
    shutil.rmtree("Training_Maps")
    os.mkdir("Training_Maps")

try:
    os.mkdir("Testing_Maps")
except FileExistsError:
    shutil.rmtree("Testing_Maps")
    os.mkdir("Testing_Maps")


perc_test = 0.1                             # Percentage of testing maps over the overall number of maps
Ntot = 3000                                 # Total number of maps
Ntrain = Ntot*(1-perc_test)                 # Number of training maps
Ntest = Ntot-Ntrain                         # Number of testing maps
N_occs = 10                                 # Number of nodes in the occupancies array definition
occupancies = np.linspace(0, 20, N_occs)    # Occupancies to be created


Ntrain_per_occ = Ntrain//N_occs             # Number of training maps for each occupancy selected
Ntest_per_occ = Ntest//N_occs               # Number of testing maps for each occupancy selected

count_train = 0                             # Counter for training maps
count_test = 0                              # Counter for testing maps

# Definition of the paths (such directories were created previously)
path_train = "Training_Maps/"
path_test = "Testing_Maps/"


for occs in occupancies:
    # Save Ntrain_per_occ maps
    for idx_train in range(int(Ntrain_per_occ)):
        mappa = Map(length=10, width=10, precision=0.1, occupancy=occs)
        mappa.save_map_as_csv(path_train, "Training", count_train)
        count_train += 1

    # Save Ntest_per_occ maps
    for idx_test in range(int(Ntest_per_occ)):
        mappa = Map(length=10, width=10, precision=0.1, occupancy=occs)
        mappa.save_map_as_csv(path_test, "Testing", count_test)
        #mappa.save_figure(count_test)
        count_test += 1
