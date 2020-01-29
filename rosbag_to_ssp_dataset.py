#!/usr/bin/env python
# System / General
from __future__ import print_function
import sys, os, glob
import pdb
# Math
import numpy as np
# Images
# from PIL import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
# ROS
import rospy, rosbag
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
# Utils
from rosbag_dataset_utils import *
from create_test_and_train_lists import *

class dataset_creater:
    def __init__(self):
        #######################################
        # USER DEFINED VALUES
        # Paths
        rosbag_dir = './rosbags'
        input_rosbag = 'rosbag_for_post_process_2019-12-18-02-10-28' + '.bag'
        self.dataset_str = 'mslquad'
        self.dataset_STR = self.dataset_str.upper()
        self.num_train_images = 1000  # will be used to call create_test_and_train_list...
        self.num_test_images = 1000  # will be used to call create_test_and_train_list...
        self.cfg_dir = './cfg/'
        
        # Topic names
        ego_quad = 'quad7'
        ado_quad = 'quad4'
        ego_pose_gt_topic = '/' + ego_quad + '/mavros/vision_pose/pose'
        ado_pose_gt_topc = '/' + ado_quad + '/mavros/vision_pose/pose'
        image_topic = '/' + ego_quad + '/camera/image_raw'
        camera_info_topic = '/' + ego_quad + '/camera/camera_info'
        self.K = None
        print("Creating dataset {}/{} with {} as ego and {} as ado".format(self.dataset_STR, self.dataset_str, ego_quad, ado_quad))
        
        # Tracking quad's camera extrinsics (from quad7.param in the msl_raptor project):
        self.tf_cam_ego = np.eye(4)
        self.tf_cam_ego[0:3, 3] = np.asarray([0.01504337, -0.06380886, -0.13854437])
        self.tf_cam_ego[0:3, 0:3] = np.reshape([-6.82621737e-04, -9.99890488e-01, -1.47832690e-02, 3.50423970e-02,  1.47502748e-02, -9.99276969e-01, 9.99385593e-01, -1.20016936e-03,  3.50284906e-02], (3, 3))
        
        # 3D tracked object info
        self.diam = 0.311
        self.bb_l = 0.27
        self.bb_w = 0.27
        self.bb_h = 0.13

        # Misc.
        self.gpus = 0
        self.num_workers =4
        #######################################

        # create dir structure
        self.dataset_dir = "./{}/{}/".format(self.dataset_STR, self.dataset_str)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.image_dir = self.dataset_dir + "images"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        self.label_dir = self.dataset_dir + "labels"
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

        # read full rosbag into lists
        print("Reading rosbag: {}...".format(input_rosbag), end=" ")
        ego_pose_msg_list = []
        ego_pose_msg_time_list = []
        ado_pose_msg_list = []
        ado_pose_msg_time_list = []
        image_msg_list = []
        time_0 = -1
        bag_data = rosbag.Bag(rosbag_dir + '/' + input_rosbag)
        for topic, msg, t in bag_data.read_messages():
            if topic == image_topic:
                if time_0 < 0:
                    time_0 = t.to_sec()
                image_msg_list.append(msg)
                pass
            elif topic == ego_pose_gt_topic:
                ego_pose_msg_list.append(msg)
                ego_pose_msg_time_list.append(t.to_sec())
                pass
            elif topic == ado_pose_gt_topc:
                ado_pose_msg_list.append(msg)
                ado_pose_msg_time_list.append(t.to_sec())
                pass
            elif self.K is None and topic == camera_info_topic:
                self.im_w = msg.width
                self.im_h = msg.height
                self.K = np.reshape(msg.K, (3, 3))
                self.dist_coefs = np.reshape(msg.D, (5,))
                self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (msg.width, msg.height), 0, (msg.width, msg.height))
        print("done reading rosbag")

        # loop through each image, find the closest corresponding data, and save the appropriate data
        self.num_imgs = len(image_msg_list)
        print("Saving {} images and their labels...".format(self.num_imgs), end= " ")
        self.bridge = CvBridge()
        for img_num, img_msg in enumerate(image_msg_list):
            t = img_msg.header.stamp.to_sec()

            # Save Label
            ego_pose_msg, _ = find_closest_by_time(t, ego_pose_msg_time_list, message_list=ego_pose_msg_list)
            ado_pose_msg, _ = find_closest_by_time(t, ado_pose_msg_time_list, message_list=ado_pose_msg_list)
            vertex_coords, max_coords, min_coords = self.vertex_coordinates(ado_pose_msg, ego_pose_msg)
            self.save_dataset_labels(img_num, vertex_coords, max_coords, min_coords)
            
            # Undistort & Save images
            image_cv = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
            image_cv = cv2.undistort(image_cv, self.K, self.dist_coefs, None, self.new_camera_matrix)
            cv2.imwrite(self.image_dir + "/{:06d}.png".format(img_num), image_cv)
        print("done saving images and labels!")

        # write config file
        print("Writing params to the ./cfg/{}.data file".format(self.dataset_str))
        self.save_cfg_file()
    

    #Given poses --> return rectangle coordinates.
    def vertex_coordinates(self, ado_pose_msg, ego_pose_msg):
        # quad dimensions to xyz coordinates in quad frame
        quad_3d_bb = np.array([[ self.bb_l/2, self.bb_l/2, self.bb_h/2, 1.],
                               [ self.bb_l/2, self.bb_l/2,-self.bb_h/2, 1.],
                               [ self.bb_l/2,-self.bb_l/2,-self.bb_h/2, 1.],
                               [ self.bb_l/2,-self.bb_l/2, self.bb_h/2, 1.],
                               [-self.bb_l/2,-self.bb_l/2, self.bb_h/2, 1.],
                               [-self.bb_l/2,-self.bb_l/2,-self.bb_h/2, 1.],
                               [-self.bb_l/2, self.bb_l/2,-self.bb_h/2, 1.],
                               [-self.bb_l/2, self.bb_l/2, self.bb_h/2, 1.]])

        # Homogeneous Transformation: ado quad -> world frame
        tf_w_ado = pose_to_tf(ado_pose_msg.pose)

        # Homogeneous Transformation: world -> ego quad frame
        tf_w_ego = pose_to_tf(ego_pose_msg.pose)
        tf_ego_w = invert_tf(tf_w_ego)

        # combined tf camera -> ado frame
        tf_cam_ado = np.matmul(np.matmul(self.tf_cam_ego, tf_ego_w), tf_w_ado)

        vertex_ado = np.zeros((4,1))
        vertex_pixels_array = np.zeros((8,2))
        for index in range(0,8):
            vertex_ado[:4,0] = quad_3d_bb[index] #column vector, (x,y,z,1)
            vertex_cam = np.matmul(tf_cam_ado, vertex_ado)
            vertex_pixels = np.matmul(self.K, vertex_cam[:-1])
            vertex_pixels = np.true_divide(vertex_pixels, vertex_pixels[2][0])  # [col, row]
            vertex_pixels_array[index,:2] = vertex_pixels[:2,0]

        vertex_pixels_array = np.squeeze(vertex_pixels_array)
        max_coords = np.amax(vertex_pixels_array,axis=0)  # [col, row]
        min_coords = np.amin(vertex_pixels_array,axis=0)  # [col, row]

        return vertex_pixels_array, max_coords, min_coords


    def save_cfg_file(self):
        filename = "{}/{}.data".format(self.cfg_dir, self.dataset_str)
        file = open(filename,'w')
        write_str = ("name = {}\ntrain = {}\nvalid = {}\nbackup = {}\n" + \
                    "diam = {}\nbox_length = {}\nbox_width = {}\nbox_height = {}\n" + \
                    "width = {}\nheight = {}\nfx = {}\nfy = {}\nu0 = {}\nv0 = {}\n" + \
                    "gpus = {}\nnum_workers = {}\n").format(
                        self.dataset_str,
                        self.dataset_STR + '/' + self.dataset_str + '/train.txt',
                        self.dataset_STR + '/' + self.dataset_str + '/test.txt',
                        'backup/' + self.dataset_str,
                        self.diam,
                        self.bb_l,
                        self.bb_w,
                        self.bb_h,
                        self.im_w,
                        self.im_h,
                        self.K[0, 0],
                        self.K[1, 1],
                        self.K[0, 2],
                        self.K[1, 2],
                        self.gpus, 
                        self.num_workers)  # mesh = MSLQUAD/mslquad/mslquad.ply
        file.write(write_str)
        file.close()
        create_test_and_train_lists(min(self.num_train_images, self.num_imgs), min(self.num_test_images, self.num_imgs), self.dataset_STR, self.dataset_str, 'png')


    def save_dataset_labels(self, img_num, vertex_coords, max_coords, min_coords):
        """
        Write a txt file for this image with the ground truth data (21 values)
        """
        # Save labels
        # 8 corners (vertex_coords) should correspond to the 
        #  [[min_x, min_y, min_z], [min_x, min_y, max_z],  [min_x, max_y, min_z], 
        #  [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z], 
        #  [max_x, max_y, min_z], [max_x, max_y, max_z]]
        vrtx_order = [5, 4, 6, 7, 2, 3, 1, 0]
        
        filename = self.label_dir + "/{:06d}.txt".format(img_num) 
        file = open(filename,'w')
        write_str = "{:d} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
            0, # 1st number: class label
            ((max_coords[0] + min_coords[0]) / 2) / self.im_w,  # 2nd number: x0 (x-coordinate of the centroid)
            ((max_coords[1] + min_coords[1]) / 2) / self.im_h,  # 3rd number: y0 (y-coordinate of the centroid)
            (vertex_coords[vrtx_order[0], 0]) / self.im_w,  # 4th number: x1 (x-coordinate of the first corner)
            (vertex_coords[vrtx_order[0], 1]) / self.im_h,  # 5th number: y1 (y-coordinate of the first corner)
            (vertex_coords[vrtx_order[1], 0]) / self.im_w,  # ...
            (vertex_coords[vrtx_order[1], 1]) / self.im_h,  # ...
            (vertex_coords[vrtx_order[2], 0]) / self.im_w,  # ...
            (vertex_coords[vrtx_order[2], 1]) / self.im_h,  # ...
            (vertex_coords[vrtx_order[3], 0]) / self.im_w,  # ...
            (vertex_coords[vrtx_order[3], 1]) / self.im_h,  # ...
            (vertex_coords[vrtx_order[4], 0]) / self.im_w,  # ...
            (vertex_coords[vrtx_order[4], 1]) / self.im_h,  # ...
            (vertex_coords[vrtx_order[5], 0]) / self.im_w,  # ...
            (vertex_coords[vrtx_order[5], 1]) / self.im_h,  # ...
            (vertex_coords[vrtx_order[6], 0]) / self.im_w,  # ...
            (vertex_coords[vrtx_order[6], 1]) / self.im_h,  # ...
            (vertex_coords[vrtx_order[7], 0]) / self.im_w,  # 18th number: x8 (x-coordinate of the eighth corner)
            (vertex_coords[vrtx_order[7], 1]) / self.im_h,  # 19th number: y8 (y-coordinate of the eighth corner)
            (max_coords[0] - min_coords[0]) / self.im_w,  # 20th number: x range
            (max_coords[1] - min_coords[1]) / self.im_h)  # 21st number: y range
        file.write(write_str)
        file.close()

if __name__ == "__main__":
    dsc = dataset_creater()
    print("Finished!")
    pdb.set_trace()
