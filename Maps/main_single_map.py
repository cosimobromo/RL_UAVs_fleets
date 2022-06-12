'''
Main Project Name:  Reinforcement Learning Exporation Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Creation of single maps --------------------------

This script allows to use the class "Map" defined in maps_creation.py and its methods to plot and save the map,
according to the specified parameters:
- length
- width
- occupancy
- precision (cell dimension)
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from maps_creation import Map

# This function allows to compute the map occupancy, subtracting borders' contribution
def comp_occ(mappa):
    border_occ = int((sz1 / prec) * 2 + (sz2 / prec) * 2 - 4)
    return (np.count_nonzero(mappa)-border_occ)/np.product(mappa.shape)*100

'''
sz1:    Dimension along x                   [m]
sz2:    Dimension along y                   [m]
prec:   Cell dimension                      [m]
occ:    Obstacle occupancy in percentage    [-]
idx:    Index for file name                 [-]
'''
sz1 = 10
sz2 = 10
prec = 0.1
occ = 20
idx = 4

# Call the Map class
mappa = Map(length = sz1, width = sz2, precision = prec, occupancy = occ)
mapp = mappa.m
path = "Case_maps/"

# Compute true occupancy (subtracting borders' contribution)
true_occ = comp_occ(mapp)

# Plot and save the map
np.savetxt(path + "Map_"+str(idx) + ".txt", mapp, delimiter = ',')
plt.figure(idx)
plt.imshow(mapp, cmap = "Greys", vmin = 0, vmax = 1)
title = 'Map with ' + str(round(true_occ)) + '% occupancy'
plt.title(title)
plt.tick_params(left=False, right=False, labelleft=False,
                labelbottom=False, bottom=False, top=False, labeltop = False)
plt.savefig(path+"Map_"+str(idx)+".png")
plt.close()