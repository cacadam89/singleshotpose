#!/usr/bin/env python3
from __future__ import print_function
import pdb
import os, sys, time
from copy import copy
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
from rosbag_dataset_utils import *

import PIL.Image
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
        self.num_classes = 1

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
            
        self.corners3D = get_3D_corners(vertices)
        # self.K = get_camera_intrinsic(u0, v0, fx, fy)

        # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
        self.model = Darknet(modelcfg)
        self.model.load_weights(weightfile)
        self.model.cuda()
        self.model.eval()
        self.shape = (self.model.test_width, self.model.test_height)
        num_keypoints = self.model.num_keypoints 
        num_labels    = num_keypoints * 2 + 3 # +2 for width, height,  +1 for class label
        ##############################################################################
        ##############################################################################
        ##############################################################################

        self.result_list = []  # save the results as they are processed
        self.itr = 0
        self.time_prev = -1
        self.bridge = CvBridge()
        self.pose_buffer_len = 20
        self.ado_pose_msg_buf = []
        self.ego_pose_msg_buf = []
        self.ado_pose_time_msg_buf = []
        self.ego_pose_time_msg_buf = []

        # Create camera (camera extrinsics from quad7.param in the msl_raptor project):
        self.tf_cam_ego = np.eye(4)
        self.tf_cam_ego[0:3, 3] = np.asarray([0.01504337, -0.06380886, -0.13854437])
        self.tf_cam_ego[0:3, 0:3] = np.reshape([-6.82621737e-04, -9.99890488e-01, -1.47832690e-02, 3.50423970e-02,  1.47502748e-02, -9.99276969e-01, 9.99385593e-01, -1.20016936e-03,  3.50284906e-02], (3, 3))
        
        # Correct Rotation w/ Manual Calibration
        Angle_x = 8./180. 
        Angle_y = 8./180.
        Angle_z = 0./180. 
        R_deltax = np.array([[ 1.             , 0.             , 0.              ],
                             [ 0.             , np.cos(Angle_x),-np.sin(Angle_x) ],
                             [ 0.             , np.sin(Angle_x), np.cos(Angle_x) ]])
        R_deltay = np.array([[ np.cos(Angle_y), 0.             , np.sin(Angle_y) ],
                             [ 0.             , 1.             , 0               ],
                             [-np.sin(Angle_y), 0.             , np.cos(Angle_y) ]])
        R_deltaz = np.array([[ np.cos(Angle_z),-np.sin(Angle_z), 0.              ],
                             [ np.sin(Angle_z), np.cos(Angle_z), 0.              ],
                             [ 0.             , 0.             , 1.              ]])
        R_delta = np.dot(R_deltax, np.dot(R_deltay, R_deltaz))
        self.tf_cam_ego[0:3,0:3] = np.matmul(R_delta, self.tf_cam_ego[0:3,0:3])
        #########################################################################################


        camera_info = rospy.wait_for_message(self.ns + '/camera/camera_info', CameraInfo, 30)
        self.K = np.reshape(camera_info.K, (3, 3))
        self.dist_coefs = np.reshape(camera_info.D, (5,))
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))
        rospy.Subscriber(self.ns + '/mavros/vision_pose/pose', PoseStamped, self.ego_pose_gt_cb, queue_size=10)  # optitrack pose
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb, queue_size=1,buff_size=2**21)
        rospy.Subscriber('/quad4' + '/mavros/vision_pose/pose', PoseStamped, self.ado_pose_gt_cb, queue_size=10)  # DEBUG ONLY - optitrack pose

        
    def ado_pose_gt_cb(self, msg):
        self.ado_pose_gt_rosmsg = msg.pose
        pose_tm = msg.header.stamp.to_sec()
        self.ado_pose_msg_buf.append(msg)
        self.ado_pose_time_msg_buf.append(pose_tm)


    def ego_pose_gt_cb(self, msg):
        self.ego_pose_gt_rosmsg = msg.pose
        pose_tm = msg.header.stamp.to_sec()
        self.ego_pose_msg_buf.append(msg)
        self.ego_pose_time_msg_buf.append(pose_tm)


    def image_cb(self, msg):
        """
        Maintains a buffer of images & times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        img_tm = msg.header.stamp.to_sec()
        img_cv2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        img_cv2 = cv2.undistort(img_cv2, self.K, self.dist_coefs, None, self.new_camera_matrix)
        img = PIL.Image.fromarray(img_cv2).resize(self.shape)

        img = transforms.ToTensor()(img).resize(1, 3, img.size[0], img.size[1]).cuda()

        img = Variable(img, volatile=True)
        output = self.model(img).data   # Forward pass

        # Using confidence threshold, eliminate low-confidence predictions
        box_pr = get_region_boxes(output, self.num_classes, self.model.num_keypoints)
        
        # Denormalize the corner predictions 
        corners2D_pr = np.array(np.reshape(box_pr[:18], [-1, 2]), dtype='float32')
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.im_width
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.im_height

        # Compute [R|t] by pnp
        R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(self.K, dtype='float32'))
        tf_cam_ado = rotm_and_t_to_tf(R_pr, t_pr)

        ado_msg, _ = find_closest_by_time(img_tm, self.ado_pose_time_msg_buf, message_list=self.ado_pose_msg_buf)
        ego_msg, _ = find_closest_by_time(img_tm, self.ego_pose_time_msg_buf, message_list=self.ego_pose_msg_buf)

        tf_w_ado_gt = pose_to_tf(ado_msg.pose)
        tf_w_ego = pose_to_tf(ego_msg.pose)

        tf_w_ado = tf_w_ego @ invert_tf(self.tf_cam_ego) @ tf_cam_ado

        quat_pr = rotm_to_quat(tf_w_ado[0:3, 0:3])
        state_pr = np.concatenate((tf_w_ado[0:3, 3], quat_pr))  # shape = (7,)

        self.result_list.append((state_pr, tf_w_ado, tf_w_ado_gt, img_tm, time.time()))
        self.itr += 1
        if self.itr > 0 and self.itr % 50 == 0:
            print("Finished processing image #{}".format(self.itr))


    def post_process_data(self):
        print("Post-processing data now ({} itrs)".format(len(self.result_list)))
        b_save_bb_imgs = True
        bb_im_path = './root/ssp_ws/src/singleshotpose/output_imgs'
        create_dir_if_missing(bb_im_path)
        for i, res in enumerate(self.result_list):
            if b_save_bb_imgs:
                draw_2d_proj_of_3D_bounding_box(img, corners2D_pr, corners2D_gt=None, epoch=None, batch_idx=None, detect_num=i, im_save_dir=bb_im_path)
            if i > 3:
                break
        print("done with post process!")


    def truths_length(self, truths, max_num_gt=50):
        for i in range(max_num_gt):
            if truths[i][1] == 0:
                return i


    def run(self):
        rate = rospy.Rate(100)
        b_flag = True
        while not rospy.is_shutdown():
            rate.sleep()
        print("1 test do we ever get here??")


if __name__ == '__main__':
    try:
        np.set_printoptions(linewidth=160, suppress=True)  # make print statments easier to read
        program = ssp_rosbag()
        program.run()
    except:
        print("2 test do we ever get here??")
        if len(self.result_list) > 0:
            self.post_process_data()
        import traceback
        traceback.print_exc()