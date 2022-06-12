'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Plot waypoints -------------------

This script is intended to plot the waypoints by reading a .plan file
'''

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import json 
import sys 

x = []
y = []
z = []

def load_file(file_name): 
	with open(file_name) as f:
	    obj = json.load(f)
	    f.close()
	d = obj['mission']
	home_pos = d['plannedHomePosition']
	if 'items' in d:
		for wp in d['items']:
			x_lat=float(wp['params'][4])
			y_long=float(wp['params'][5])
			z_alt=float(wp['params'][6])
			x.append(x_lat) 
			y.append(y_long) 
			z.append(z_alt) 
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(x,y,z) 
	#ax.scatter(home_pos[0], home_pos[1], home_pos[2], marker = 'x') 
	#ax.text(home_pos[0], home_pos[1], home_pos[2], "Home position")
	
	for i, point in enumerate(x): 
		ax.text(x[i], y[i], z[i], str(i+1))
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	plt.show()
	

if __name__ == '__main__': 
	if len(sys.argv) > 1: 
		file_name = sys.argv[1]
		load_file(file_name)
	else: 
		print("No file selected") 
		
