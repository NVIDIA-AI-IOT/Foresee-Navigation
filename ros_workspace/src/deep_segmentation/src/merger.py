import copy
import rospy
import thread
import math
from sensor_msgs.msg import LaserScan


class Merger:
    def __init__(self, rplidar_topic, pub_topic):
        rospy.Subscriber(rplidar_topic, LaserScan, self._callback)
        self.rplidar = LaserScan()
        self.cam_scan = LaserScan()
        self.rplock = thread.allocate_lock()
        # self.camlock = thread.allocate_lock()
        self.merge_pub = rospy.Publisher(
            pub_topic, LaserScan, queue_size=10)
        # self.should_continue = True
        self.seq = self.rplidar.header.seq

    # def __del__(self):
        # self.should_continue = False

    def _callback(self, data):
        self.rplock.acquire()
        if data.header.seq > self.seq:
            self.seq = data.header.seq
        else:
            self.rplock.release()
            return
        self.rplock.release()

        rplidar = data
        rplidar.ranges = list(rplidar.ranges)
        for idx, p in enumerate(self.cam_scan.ranges):
            mapped_ind = self._map_index(
                idx, self.cam_scan.angle_min, self.cam_scan.angle_increment, rplidar.angle_min, rplidar.angle_increment)
            if mapped_ind >= len(rplidar.ranges):
                continue
            rplidar.ranges[mapped_ind] = p
        self.merge_pub.publish(rplidar)

    def provide_cam(self, scan):
        # self.camlock.acquire()
        # self.cam_scan = copy.deepcopy(scan)
        # self.camlock.release()
        self.cam_scan = scan

    @staticmethod
    def _map_index(src_ind, src_min, src_inc, dst_min, dst_inc):
        angle = src_ind * src_inc + src_min + math.radians(90)
        while angle > math.pi:
            angle -= 2*math.pi
        while angle < -math.pi:
            angle += 2*math.pi
        return int(math.floor((angle - dst_min)/dst_inc))

    # def thread_func(self):
    #     while self.should_continue:
    #         self.rplock.acquire()
    #         rplidar = copy.deepcopy(self.rplidar)
    #         self.rplock.release()
    #         rplidar.ranges = list(rplidar.ranges)
