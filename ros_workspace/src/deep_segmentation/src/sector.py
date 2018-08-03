#!/usr/bin/env python
import numpy as np
import cv2
import math
from time import sleep
INF = float("inf")
class PCLGen:
	def __init__(self, im_shape, num_sectors=25):
		"do nothing"
		points = self._get_sectors_points(im_shape, num_sectors)
		transformed_pts = []
		for point in points:
			transformed_pt = (int(point[0]), 500-int(point[1]))
			transformed_pts.append(transformed_pt)
		self.masks = self._make_sectors(transformed_pts, (im_shape[1]/2, im_shape[0]), im_shape)
		
	@staticmethod
	def _get_sectors_points(im_shape, sectors):
		height = im_shape[0]
		width = im_shape[1]
		theta_ang = 180.0/sectors
		points_list = []
		unique_ang = 0
		points_list.append((0, 0))
		for i in xrange(1, sectors+1):
			if unique_ang < 90:
				unique_ang = unique_ang + theta_ang
				y_coord = 0.5*width*math.tan(math.radians(unique_ang))
				x_coord = 0
				if y_coord > height or y_coord < 0:
					slope = y_coord/(0.5*width)
					x_coord = (y_coord-height)/slope
					y_coord = height
				points_list.append((x_coord, y_coord))
		
		length_of = len(points_list) 
		for i in xrange(length_of):
			old_points = points_list[length_of-i-1]
			diff = width/2 - old_points[0]
			points_list.append((old_points[0] + 2*diff, old_points[1]));
		return points_list

	@staticmethod
	def _make_sectors(points_list, center_point, size):
		masks = []
		for i in xrange(len(points_list) - 1):
			mask = np.zeros(size, dtype=np.uint8)
			mask = cv2.fillPoly(mask, np.array([[center_point, points_list[i], points_list[i+1]]], dtype=np.int32), 255)
			masks.append(mask)
		return masks	
		
		
	def find_points(self, img):
		pts = []
		for mask in self.masks:
			_mask = cv2.bitwise_and(mask, img)
			img2, contours, _ = cv2.findContours(_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			if len(contours) < 1:
                            pts.append((INF, INF))
                        else:
                            cnt = contours[0]
			    pts.append(tuple(cnt[cnt[:,:,1].argmax()][0]))
		return pts
