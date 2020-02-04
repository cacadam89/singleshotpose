#!/usr/bin/env python3
import pdb
import os, sys
import time
import torch
import scipy.io
import warnings
from torch.autograd import Variable
from torchvision import datasets, transforms

from darknet import Darknet
from utils import *
from MeshPly import MeshPly

from raptor_specific_utils import *

import cv2
from cv_bridge import CvBridge, CvBridgeError
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

class ssp_rosbag:
    def __init__(self):
        rospy.init_node('eval_baseline', anonymous=True)

        ##############################################################################
        ##############################################################################
        ##############################################################################
        self.ns = rospy.get_param('~ns')  # robot namespace
        modelcfg = rospy.get_param('~modelcfg')
        weightfile = rospy.get_param('~weightfile')
        datacfg = rospy.get_param('~datacfg')

        # Parse configuration files
        data_options = read_data_cfg(datacfg)
        valid_images = data_options['valid']

        if 'mesh' in data_options:
            meshname  = data_options['mesh']
        else:
            meshname  = None
            assert('box_length' in data_options)
            box_length = float(data_options['box_length'])
            box_width = float(data_options['box_width'])
            box_height = float(data_options['box_height'])

        name         = data_options['name']
        gpus         = data_options['gpus'] 
        self.im_width     = int(data_options['width'])
        self.im_height    = int(data_options['height'])

         # Parameters
        seed = int(time.time())
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
        num_classes = 1

        # Read object model information, get 3D bounding box corners
        if meshname is None:
            # vertices must be 4 x N for compute_projections to work later
            vertices = np.array([[ box_length/2, box_width/2, box_height/2, 1.],
                                [ box_length/2, box_width/2,-box_height/2, 1.],
                                [ box_length/2,-box_width/2,-box_height/2, 1.],
                                [ box_length/2,-box_width/2, box_height/2, 1.],
                                [-box_length/2,-box_width/2, box_height/2, 1.],
                                [-box_length/2,-box_width/2,-box_height/2, 1.],
                                [-box_length/2, box_width/2,-box_height/2, 1.],
                                [-box_length/2, box_width/2, box_height/2, 1.]]).T
            diam  = float(data_options['diam'])
        else:
            mesh             = MeshPly(meshname)
            vertices         = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
            try:
                diam  = float(data_options['diam'])
            except:
                diam  = calc_pts_diameter(np.array(mesh.vertices))
            
        self.corners3D            = get_3D_corners(vertices)
        # self.K = get_camera_intrinsic(u0, v0, fx, fy)

        # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
        self.model = Darknet(modelcfg)
        self.model.print_network()
        self.model.load_weights(weightfile)
        self.model.cuda()
        self.model.eval()
        self.shape = (self.model.test_width, self.model.test_height)
        num_keypoints = self.model.num_keypoints 
        num_labels    = num_keypoints * 2 + 3 # +2 for width, height,  +1 for class label
        ##############################################################################
        ##############################################################################
        ##############################################################################

        self.itr = 0
        self.bridge = CvBridge()
        self.img_buffer = ([], [])
        self.img_rosmesg_buffer_len = 10


        camera_info = rospy.wait_for_message(self.ns + '/camera/camera_info', CameraInfo, 30)
        self.K = np.reshape(camera_info.K, (3, 3))
        self.dist_coefs = np.reshape(camera_info.D, (5,))
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 1, (camera_info.width, camera_info.height))
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb, queue_size=1,buff_size=2**21)
        

    def image_cb(self, msg):
        """
        Maintains a buffer of images & times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """

        # im_msg = self.find_closest_by_time_ros2(my_time, self.img_buffer[1], self.img_buffer[0])[0]
        image = self.bridge.imgmsg_to_cv2(im_msg, desired_encoding="passthrough")
        img = cv2.undistort(image, self.K, self.dist_coefs, None, self.new_camera_matrix)
        
        # img = Image.open(imgpath).convert('RGB')
        img = img.resize(self.shape)
        img = img.cuda()
        img = Variable(img, volatile=True)
        
        output = self.model(data).data   # Forward pass
        
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)   
        
        # Denormalize the corner predictions 
        corners2D_pr = np.array(np.reshape(box_pr[:18], [-1, 2]), dtype='float32')
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.im_width
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.im_height
        # preds_corners2D.append(corners2D_pr)

         # [OPTIONAL] generate images with bb drawn on them
        # draw_2d_proj_of_3D_bounding_box(data, corners2D_pr, corners2D_gt, None, batch_idx, k, im_save_dir = "./backup/{}/valid_output_images/".format(name))
        
        # Compute [R|t] by pnp
        R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(self.K, dtype='float32'))
        
        self.itr += 1


    def truths_length(self, truths, max_num_gt=50):
        for i in range(max_num_gt):
            if truths[i][1] == 0:
                return i

    def find_closest_by_time_ros2(self, time_to_match, time_list, message_list=None):
        """
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

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    try:
        program = ssp_rosbag()
        program.run()
    except:
        import traceback
        traceback.print_exc()