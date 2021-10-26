#! /usr/bin/python

import numpy as np
import rospy
import time
from numpy import genfromtxt
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

CELL_OCC = 0.9
CELL_FREE = 0.01
CELL_UNKNOWN = 0.45

BASE_LINK_FRAME = 'base_link'
MAP_RESOLUTION = 0.1
MAP_SIZE_X = 20.0
MAP_SIZE_Y = 20.0
MAP_CENTER_X = -10.0
MAP_CENTER_Y = -10.0
MAP_ROWS = int(MAP_SIZE_Y/MAP_RESOLUTION)
MAP_COLS = int(MAP_SIZE_X/MAP_RESOLUTION)


class OccGrid:
    def __init__(self):
        rospy.init_node("OccupancyGrid")

        self.msg = OccupancyGrid()
        self.msg.header.frame_id = BASE_LINK_FRAME
        self.msg.info.resolution = MAP_RESOLUTION
        self.msg.info.width = MAP_COLS
        self.msg.info.height = MAP_ROWS
        self.msg.info.origin.position.x = MAP_CENTER_X
        self.msg.info.origin.position.y = MAP_CENTER_Y

        self.grid = self.log(CELL_UNKNOWN) * np.ones((MAP_ROWS, MAP_COLS))

        self.sub = rospy.Subscriber('base_scan', LaserScan, self.create_grid, queue_size=1)
        self.pub = rospy.Publisher('map', OccupancyGrid, queue_size = 1)

	#self.ranges = genfromtxt('data_ranges.csv', delimiter=',')
	#self.create_grid()

    # p(x) = 1 - \frac{1}{1 + e^l(x)}
    def prob(self, log):
        res = np.exp(log) / (1.0 + np.exp(log))
        return res

    # l(x) = log(\frac{p(x)}{1 - p(x)})
    def log(self, prob):
        return np.log(prob/(1-prob))

    def is_inside (self, i, j):
        return i<self.grid.shape[0] and j<self.grid.shape[1] and i>=0 and j>=0

    def create_grid(self, data):
        angle_min = data.angle_min
        angle_inc = data.angle_increment

        for idx, value in enumerate(data.ranges):
            theta = angle_min + idx * angle_inc
            x0 = -MAP_CENTER_X / MAP_RESOLUTION
            y0 = -MAP_CENTER_Y / MAP_RESOLUTION
            x1 = (value*np.cos(theta)-MAP_CENTER_X) / MAP_RESOLUTION
            y1 = (value*np.sin(theta)-MAP_CENTER_Y) / MAP_RESOLUTION
            d_cells = value / MAP_RESOLUTION
            ip, jp = self.bresenham(y0, x0, y1, x1, d_cells)
            self.grid[int(ip), int(jp)] += self.log(CELL_OCC) - self.log(CELL_UNKNOWN)

        occ_grid = self.grid.flatten()
        self.publish_grid(occ_grid)

    def bresenham(self, x0, y0, x1, y1, d):
        dx = abs(y1 - y0)
        sx = 1 if y0 < y1 else -1

        dy = -1 * abs(x1 - x0)
        sy = 1 if x0 < x1 else -1

        jp, ip = y0, x0
        err = dx+dy

        while True:
            if (jp == y1 and ip == x1) or (np.sqrt((jp - y0) ** 2 + (ip - x0) ** 2) >= d) or not self.is_inside(ip, jp):
                return ip, jp
            elif self.grid[int(ip),int(jp)]==100:
                return ip, jp

            if self.is_inside(ip, jp):
                self.grid[int(ip),int(jp)] += self.log(CELL_FREE) - self.log(CELL_UNKNOWN)

            e2 = 2*err
            if e2 >= dy:
                err += dy
                jp += sx
            if e2 <= dx:
                err += dx
                ip += sy

    def publish_grid(self, occ_grid):
        probability_map = (self.prob(occ_grid) * 100).astype(dtype=np.int8)
        
        self.msg.data = probability_map
        self.pub.publish(self.msg)
	rospy.loginfo("!")
	#to do it once
	rate = rospy.Rate(0.01)
        rate.sleep()

occupancy_grid = OccGrid()
rospy.spin()
