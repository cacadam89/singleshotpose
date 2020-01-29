####### ROS UTILITIES #######
# IMPORTS
# system
import pdb
# math
import numpy as np
import numpy.linalg as la
from bisect import bisect_left
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from scipy.spatial.transform import Rotation as R

def pose_to_tf(pose):
    """
    tf_w_q (w:world, q:quad) s.t. if a point is in the quad frame (p_q) then
    the point transformed to be in the world frame is p_w = tf_w_q * p_q.
    """
    tf_w_q = quat_to_tf([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    tf_w_q[0:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
    return tf_w_q


def quat_to_rotm(quat):
    """ 
    calculate the rotation matrix of a given quaternion (frames assumed to be consistant 
    with the UKF state quaternion). First element of quat is the scalar.
    """
    return R.from_quat(np.roll(np.reshape(quat, (-1, 4)),3,axis=1)).as_dcm()


def quat_to_tf(quat):
    """ 
    calculate the rotation matrix of a given quaternion (frames assumed to be consistant 
    with the UKF state quaternion). First element of quat is the scalar.
    """
    tf_out = np.eye(4)
    tf_out[0:3, 0:3] = quat_to_rotm(quat)
    return tf_out


def invert_tf(tf):
    tf[0:3, 0:3] = tf[0:3, 0:3].T
    tf[0:3, 3] = np.matmul(-tf[0:3, 0:3], tf[0:3, 3])
    return tf


def find_closest_by_time(time_to_match, time_list, message_list=None):
    """
    time_to_match : time we want to get the closest match to [float]
    time_list : list of times [list<floats>]
    message_list : list of messages [list<ros msg>]
    Assumes lists are sorted earlier to later. Returns closest item in list by time. If two numbers are equally close, return the smallest number.
    Adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    """
    if not message_list:
        message_list = time_list
    pos = bisect_left(time_list, time_to_match)
    if pos == 0:
        return message_list[0], 0
    if pos == len(time_list):
        return message_list[-1], len(message_list) - 1
    before = time_list[pos - 1]
    after = time_list[pos]
    if after - time_to_match < time_to_match - before:
       return message_list[pos], pos
    else:
       return message_list[pos - 1], pos - 1