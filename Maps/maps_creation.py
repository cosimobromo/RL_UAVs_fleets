'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Map class -------------------

This script is intended for the creation of a class "Map" used for:
- map creation with randomly distributed obstacles
- map plotting
- map saving
'''

import matplotlib.colors
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Map:
    def __init__(self, length=100, width=100, precision=0.1, occupancy=0):
        '''
        Initialization of the Map object, with the following input parameters:
            length:         Dimension n. 1 [m]
            width:          Dimension n. 2 [m]
            precision:      Dimension of 1 cell [m]
            occupancy:      Percentage of obstacles on the overall map [-]
        '''
        self.a = length
        self.b = width
        self.step = precision
        self.occupancy = occupancy

        # Definition of the matrix map according to the specificed dimensions and precision
        sz1 = int(self.a/self.step)
        sz2 = int(self.b/self.step)
        self.m = np.zeros((int(sz1), int(sz2)))

        # At first, the map has no obstacles placed
        occ = 0;        # Initialization of the obstacle current occupation

        # Initialize lists containing, for each obstacle, center and dimensions
        centers = []
        dims = []

        # Until obstacle threshold has not been reached
        while occ < self.occupancy:
            # Select a specific coordinate to define a vertex of the rectangular obstacle
            x_obs = np.random.randint(int(sz1))
            y_obs = np.random.randint(int(sz2))

            '''
            Now select randomly some dimensions for the obstacle, such that obstacle dimension in each direction 
            is between 8 % and 20 % of the map size along that dimension
            '''
            max_dim_x = int(np.floor(sz1*0.20))
            max_dim_y = int(np.floor(sz1*0.20))
            min_dim_x = int(np.floor(sz1*0.08))
            min_dim_y = int(np.floor(sz1*0.08))
            dim_x = np.random.randint(min_dim_x, max_dim_x)
            dim_y = np.random.randint(min_dim_y, max_dim_y)

            '''
            If the obstacle is near to one of the map borders, fill the border itself, too 
            '''
            if x_obs <= np.ceil(sz1*0.08):
                x_obs = 0
            if y_obs <= np.ceil(sz2*0.08):
                y_obs = 0
            if sz1-(x_obs+dim_x) <= np.ceil(sz1 * 0.08):
                dim_x = int(sz1-x_obs)
            if sz2-(y_obs + dim_y) <= np.ceil(sz2 * 0.08):
                dim_y = int(sz2-y_obs)

            # Compute current obstacle center
            current_center = np.array([x_obs+dim_x/2, y_obs+dim_y/2])
            flag = 0        # For placeability of objects
            if occ == 0:    # No obstacle has been placed
                centers.append([x_obs + dim_x / 2, y_obs + dim_y / 2])
                dims.append([dim_x, dim_y])
            else:
                for idx, center in enumerate(centers):
                    # Check if the new obstacle is placeable, i.e. two obstacles' do not overlap,
                    # with a given safety factor (1.55)
                    if np.linalg.norm(center-current_center, np.inf) <= max([dim_x, dim_y]+dims[idx])*1.55:
                        flag = 1    # Not placeable object
                        break


            # If obstacle is placeable, place it
            if flag == 0:
                centers.append([x_obs + dim_x / 2, y_obs + dim_y / 2])
                dims.append([dim_x, dim_y])
                # Place the obstacle
                self.m[x_obs:x_obs+dim_x, y_obs:y_obs+dim_y] = int(1);
                occ = occ + (dim_x * dim_y) / (sz1 * sz2) * 100;
            else:
                pass # Do nothing, ignore the current obstacle created

        # Set all borders as an obstacle --> UAVs should not go out of the map
        self.m[0,:] = 1
        self.m[-1,:] = 1
        self.m[:,0] = 1
        self.m[:,-1] = 1
        self.occ = occ + (2*(sz1+sz2)-4)/(sz1 * sz2)*100
        # Compute final occupancy of the map
        self.m = np.array(self.m, dtype = 'int')

    '''
    show_image function allows to plot interactively the currently created map 
    '''
    def show_image(self):
        plt.matshow(self.m, cmap = 'Greens')
        title = str(self.a) + 'm  x ' +str(self.b) + 'm map with ' + str(round(self.occ,1)) + '% occupancy'
        plt.title(title)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False, top=False, labeltop = False)
        plt.show()


    '''
    save_figure function allows to save the plotted map as a .png file
    '''
    def save_figure(self, idx):
        plt.matshow(self.m, cmap='Greens')
        title = str(self.a) + 'm  x ' +str(self.b) + 'm map with ' + str(round(self.occ,1)) + '% occupancy'
        plt.title(title)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False, top=False, labeltop = False)
        plt.savefig("Photos/Map_"+str(idx)+".png")
        plt.close()

    '''
    save_map_as_csv allows to save the plotted map as a .txt file (used later for training)
    '''
    def save_map_as_csv(self, p, label, index):
        filename = p + label + '_map_'+str(index)
        np.savetxt(filename, self.m, delimiter=",")
