'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Waypoints to coordinates -------------------

This script is intended to convert the waypoints read from a .csv file into a flight plan in .plan format (json), converting them in latitude and longitude coordinates. 
'''

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import json 
import sys 
import pandas as pd 
import numpy as np


x = []
y = []
z = []

conv = 9e-6

def load_file(file_name, plot_flag = False): 
	d = pd.read_csv(file_name, names = ['x', 'y'])
	x = d['x'].values
	y = d['y'].values
	
	if plot_flag: 
		fig, ax = plt.subplots()
		ax.plot(x,y) 

		for i, point in enumerate(x): 
			ax.text(x[i], y[i], str(i+1))
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		plt.show()
	return x, y 
	
def gps_traj(long_start, lat_start, x, y): 
	lats = []
	longs = []
	for i, _ in enumerate(x): 
		lats.append(long_start + conv*y[i])
		longs.append(lat_start + conv/np.cos(lats[-1]*np.pi/180)*x[i])
	return lats, longs
	
def plot_gps_traj(lats, longs): 
	fig, ax = plt.subplots() 
	ax.plot(longs, lats) 
	plt.show()
	
def write_plan(lats, longs, idx): 
	data = {"fileType": "Plan", "geoFence": {"circles": [], "polygons": [], "version": 2}, "groundStation": "QGroundControl", "rallyPoints": {"points": [], "version": 2},"version": 1}
	data['mission'] = {"cruiseSpeed": 15, "firmwareType": 12, "hoverSpeed": 5, "plannedHomePosition": [47.397742, 8.545594, 488], "vehicleType": 2, "version": 2}
	data['mission']['items'] = []
	for i, _ in enumerate(lats): 
        	# Select altitude
        	if i == 0: 
        		alt = 5
        	elif i != len(lats)-1: 
        		alt = 10
        	else: 
        		alt = 0
        		
        	# Select command 
        	if i == 0: 
        		com = 22
        	elif i != len(lats)-1: 
        		com = 16
        	else: 
        		com = 21
        	
        	
        	data['mission']['items'].append({
                "AMSLAltAboveTerrain": None,
                "Altitude": alt,
                "AltitudeMode": 0,
                "autoContinue": True,
                "command": com,
                "doJumpId": i+1,
                "frame": 3,
                "params": [
                    0,
                    0,
                    0,
                    None,
                    lats[i],
                    longs[i],
                    alt
                ],
                "type": "SimpleItem"
            })
	
	with open('Traj_'+str(idx)+'.plan', 'w') as f:
		json.dump(data, f, ensure_ascii=False)

if __name__ == '__main__': 
	if len(sys.argv) > 2: 
		file_name = sys.argv[1]
		x, y = load_file(file_name)
		long_start = 47.39773941040039
		lat_start = 8.5455904006958
		lats, longs = gps_traj(long_start, lat_start, x, y) 
		#plot_gps_traj(lats, longs)
		write_plan(lats, longs, sys.argv[2]) 
	else: 
		print("No sufficient info") 
		
