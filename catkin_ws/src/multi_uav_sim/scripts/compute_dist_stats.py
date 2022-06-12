#!/usr/bin/env python3

'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Computation of statistics distances -------------------

This script is intended to initialize a node that reads the mutual distances among UAVs and published minimum and average mutual distance.
'''
from __future__ import division
PKG = 'px4'
import rospy
import glob
import json
import math
import os
from px4tools import ulog
import sys
from mavros import mavlink
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from mavros_msgs.msg import Mavlink, Waypoint, WaypointReached
from mavros_test_common_1 import MavrosTestCommon
from pymavlink import mavutil
from six.moves import xrange
from threading import Thread
import numpy as np

pub_min = rospy.Publisher('min_dist', Float64, queue_size=10)
pub_avg = rospy.Publisher('avg_dist', Float64, queue_size=10)

d = [[],[],[],[],[],[]]

def d_01(data): 
    d[0].append(data.data)
    min_idx = min([len(d[i]) for i in range(len(d))])
    if min_idx > 0: 
	    min_dist = min([d[j][min_idx-1] for j in range(6)])
	    avg_dist = np.mean([d[j][min_idx-1] for j in range(6)])
	    pub_min.publish(min_dist) 
	    pub_avg.publish(avg_dist)
    	    
def d_02(data): 
    d[1].append(data.data)
def d_03(data): 
    d[2].append(data.data)    
def d_12(data): 
    d[3].append(data.data)
def d_13(data): 
    d[4].append(data.data)
def d_23(data): 
    d[5].append(data.data)  
    
      
def dists_listener(): 
    rospy.Subscriber('d_01', Float64, d_01)
    rospy.Subscriber('d_02', Float64, d_02)
    rospy.Subscriber('d_03', Float64, d_03)
    rospy.Subscriber('d_12', Float64, d_12)
    rospy.Subscriber('d_13', Float64, d_13)
    rospy.Subscriber('d_23', Float64, d_23)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('compute_dist_stats', anonymous=True)
    dists_listener()
