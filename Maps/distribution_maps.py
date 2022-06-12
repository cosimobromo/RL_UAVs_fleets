'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Maps distribution --------------------------

This script allows to plot the distribution of obstacle occupancy in the maps 
'''

import numpy as np
import os
import shutil
from maps_creation import Map
import matplotlib.pyplot as plt 


plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

folder_name = "Testing_Maps/"

n_cells = 100*100
a_priori = 396
oo_dist = []
N_maps = 300
for idx in range(N_maps): 
	name = folder_name+"Testing_map_"+str(idx) 
	mappa = np.genfromtxt(name, delimiter = ',') 
	oo = 100*(np.count_nonzero(mappa)-a_priori)/n_cells
	oo_dist.append(oo) 
	
oo_dist = np.array(oo_dist)	
fig, axs = plt.subplots(1,1)

n_bins = 10
axs.hist(oo_dist, bins=n_bins, edgecolor='red', facecolor = "orange")
axs.set_xlabel("Obstacle occupancy [\%]")
axs.set_ylabel("Frequency")
axs.set_title("Percentage of obstacle presence in test maps") 
plt.savefig("Distribution_maps.eps")
plt.show()
