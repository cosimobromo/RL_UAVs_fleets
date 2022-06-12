#!/usr/bin/env python3
'''
Main Project Name:  Reinforcement Learning Exploration Strategies for UAVs fleet
Author:             Cosimo Bromo
University:         Politecnico di Torino

------------------- Computation of distances -------------------

This script is intended to initialize a node that reads the poses of the UAVs in the fleet and publishes mutual distances among them.
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

x0 = [[], [], [], []]
x1 = [[], [], [], []]
x2 = [[], [], [], []]
x3 = [[], [], [], []]

pub_01 = rospy.Publisher('d_01', Float64, queue_size=10)
pub_02 = rospy.Publisher('d_02', Float64, queue_size=10)
pub_03 = rospy.Publisher('d_03', Float64, queue_size=10)
pub_12 = rospy.Publisher('d_12', Float64, queue_size=10)
pub_13 = rospy.Publisher('d_13', Float64, queue_size=10)
pub_23 = rospy.Publisher('d_23', Float64, queue_size=10) 
pub_avg = rospy.Publisher('avg_dist', Float64, queue_size=10) 
pub_min = rospy.Publisher('min_dist', Float64, queue_size=10) 

def euc_dist(x1,y1,z1, x2,y2,z2): 
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 +0* (z1-z2)**2)

def pos_0(data): 
    x0[0].append(data.pose.position.x)
    x0[1].append(data.pose.position.y)
    x0[2].append(data.pose.position.z)
    x0[3].append(data.header.stamp)
    if len(x1[0]) != 0: 
    	pub_01.publish(euc_dist(x0[0][-1], x0[1][-1], x0[2][-1], x1[0][-1], x1[1][-1], x1[2][-1]))
    if len(x2[0]) != 0: 
    	pub_02.publish(euc_dist(x0[0][-1], x0[1][-1], x0[2][-1], x2[0][-1], x2[1][-1], x2[2][-1]))
    if len(x3[0]) != 0: 
    	pub_03.publish(euc_dist(x0[0][-1], x0[1][-1], x0[2][-1], x3[0][-1], x3[1][-1], x3[2][-1]))
    
def pos_1(data): 
    x1[0].append(data.pose.position.x)
    x1[1].append(data.pose.position.y)
    x1[2].append(data.pose.position.z)
    x1[3].append(data.header.stamp)
    if len(x2[0]) != 0: 
    	pub_12.publish(euc_dist(x1[0][-1], x1[1][-1], x1[2][-1], x2[0][-1], x2[1][-1], x2[2][-1]))
    if len(x2[0]) != 0: 
    	pub_13.publish(euc_dist(x1[0][-1], x1[1][-1], x1[2][-1], x3[0][-1], x3[1][-1], x3[2][-1]))

def pos_2(data): 
    x2[0].append(data.pose.position.x)
    x2[1].append(data.pose.position.y)
    x2[2].append(data.pose.position.z)
    x2[3].append(data.header.stamp)
    if len(x3[0]) != 0: 
    	pub_23.publish(euc_dist(x2[0][-1], x2[1][-1], x2[2][-1], x3[0][-1], x3[1][-1], x3[2][-1]))

def pos_3(data): 
    x3[0].append(data.pose.position.x)
    x3[1].append(data.pose.position.y)
    x3[2].append(data.pose.position.z)
    x3[3].append(data.header.stamp)

def poses_listener(): 
    rospy.Subscriber('uav0/mavros/local_position/pose', PoseStamped, pos_0)
    rospy.Subscriber('uav1/mavros/local_position/pose', PoseStamped, pos_1)
    rospy.Subscriber('uav2/mavros/local_position/pose', PoseStamped, pos_2)
    rospy.Subscriber('uav3/mavros/local_position/pose', PoseStamped, pos_3) 
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('compute_distances', anonymous=True)
    poses_listener()
