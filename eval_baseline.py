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
        self.b_first_rb_loop = True
        self.first_time = None

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

        self.name         = data_options['name']
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
            self.diam  = float(data_options['diam'])
        else:
            mesh             = MeshPly(meshname)
            vertices         = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
            try:
                self.diam  = float(data_options['diam'])
            except:
                self.diam  = calc_pts_diameter(np.array(mesh.vertices))
        self.vertices = vertices    
        self.corners3D = get_3D_corners(vertices)

        # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
        torch.set_grad_enabled(False)  # since we are just doing forward passes
        self.model = Darknet(modelcfg)
        self.model.load_weights(weightfile)
        self.model.cuda()
        self.model.eval()
        self.shape = (self.model.test_width, self.model.test_height)
        num_keypoints = self.model.num_keypoints 
        num_labels = num_keypoints * 2 + 3 # +2 for width, height,  +1 for class label
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
        if self.first_time is not None and self.first_time >= msg.header.stamp.to_sec():
            return
        self.ado_pose_gt_rosmsg = msg.pose
        pose_tm = msg.header.stamp.to_sec()
        self.ado_pose_msg_buf.append(msg)
        self.ado_pose_time_msg_buf.append(pose_tm)


    def ego_pose_gt_cb(self, msg):
        if self.first_time is not None and self.first_time >= msg.header.stamp.to_sec():
            return
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
        if len(program.result_list) > 0 and img_tm <= self.result_list[-1][5]:
            return

        img_cv2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        img_cv2 = cv2.undistort(img_cv2, self.K, self.dist_coefs, None, self.new_camera_matrix)
        img_pil = PIL.Image.fromarray(img_cv2).resize(self.shape)

        img =  Variable(transforms.ToTensor()(img_pil).resize(1, 3, img_pil.size[0], img_pil.size[1]).cuda(), volatile=True)

        with torch.no_grad():
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

        tf_w_cam = tf_w_ego @ invert_tf(self.tf_cam_ego)

        tf_w_ado = tf_w_cam @ tf_cam_ado

        quat_pr = rotm_to_quat(tf_w_ado[0:3, 0:3])
        state_pr = np.concatenate((tf_w_ado[0:3, 3], quat_pr))  # shape = (7,)

        img_to_save = copy(np.array(img.cpu()))
        ############################
        ############################
        # bb_im_path = os.path.dirname(os.path.relpath(__file__)) + '/output_imgs' # PATH MUST BE RELATIVE
        # create_dir_if_missing(bb_im_path)
        # tf_cam_ado_gt = invert_tf(tf_w_cam) @ tf_w_ado_gt
        # R_gt = tf_cam_ado_gt[0:3, 0:3]
        # t_gt = tf_cam_ado_gt[0:3, 3].reshape(t_pr.shape)
        # Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
        # corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), self.vertices)), Rt_gt, self.new_camera_matrix).T
        # draw_2d_proj_of_3D_bounding_box(img_to_save, corners2D_pr, corners2D_gt=corners2D_gt, epoch=None, batch_idx=None, detect_num=self.itr, im_save_dir=bb_im_path)

        # print("itr {}\n{}\n{}\n{}\n{}\n".format(self.itr, self.tf_cam_ego, invert_tf(self.tf_cam_ego), tf_w_ego, tf_w_cam))
        self.tf_cam_ego

        # pdb.set_trace()
        ############################
        ############################

        self.result_list.append((state_pr, copy(tf_w_ado), copy(tf_w_ado_gt), copy(corners2D_pr), img_to_save, img_tm, time.time(), copy(R_pr), copy(t_pr), invert_tf(tf_w_cam), copy(tf_w_ego)) )
        del img
        self.itr += 1
        if self.itr > 0 and self.itr % 50 == 0:
            print("Finished processing image #{}".format(self.itr))
            torch.cuda.empty_cache()


    def post_process_data(self):
        print("Post-processing data now ({} itrs)".format(len(self.result_list)))
        b_save_bb_imgs = True
        bb_im_path = os.path.dirname(os.path.relpath(__file__)) + '/output_imgs' # PATH MUST BE RELATIVE
        create_dir_if_missing(bb_im_path)
        N = len(self.result_list)

        # To save
        trans_dist = 0.0
        angle_dist = 0.0
        pixel_dist = 0.0
        testing_samples = 0.0
        testing_error_trans = 0.0
        testing_error_angle = 0.0
        testing_error_pixel = 0.0
        errs_2d             = []
        errs_3d             = []
        errs_trans          = []
        errs_angle          = []
        errs_corner2D       = []
        preds_trans         = []
        preds_rot           = []
        preds_corners2D     = []
        gts_trans           = []
        gts_rot             = []
        gts_corners2D       = []
        corners2D_gt = None
        for i, res in enumerate(self.result_list):

            # extract /  compute values for comparison
            state_pr, tf_w_ado, tf_w_ado_gt, corners2D_pr, img, img_tm, sys_time, R_pr, t_pr, tf_cam_w, tf_w_ego = res
            tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt
            R_gt = tf_cam_ado_gt[0:3, 0:3]
            t_gt = tf_cam_ado_gt[0:3, 3].reshape(t_pr.shape)

            # Compute translation error
            trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
            errs_trans.append(trans_dist)
            
            # Compute angle error
            angle_dist = calcAngularDistance(R_gt, R_pr)
            errs_angle.append(angle_dist)
            
            # Compute pixel error
            Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
            Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
            proj_2d_gt   = compute_projection(self.vertices, Rt_gt, self.new_camera_matrix)
            proj_2d_pred = compute_projection(self.vertices, Rt_pr, self.new_camera_matrix) 
            norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
            pixel_dist   = np.mean(norm)
            errs_2d.append(pixel_dist)

            # Compute corner prediction error
            corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), self.vertices)), Rt_gt, self.new_camera_matrix).T
            corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
            corner_dist = np.mean(corner_norm)
            errs_corner2D.append(corner_dist)

            # Compute 3D distances
            transform_3d_gt   = compute_transformation(self.vertices, Rt_gt) 
            transform_3d_pred = compute_transformation(self.vertices, Rt_pr)  
            norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
            vertex_dist       = np.mean(norm3d)
            errs_3d.append(vertex_dist)

            # Sum errors
            testing_error_trans  += trans_dist
            testing_error_angle  += angle_dist
            testing_error_pixel  += pixel_dist
            testing_samples      += 1
            if b_save_bb_imgs:
                draw_2d_proj_of_3D_bounding_box(img, corners2D_pr, corners2D_gt=corners2D_gt, epoch=None, batch_idx=None, detect_num=i, im_save_dir=bb_im_path)

        # Compute 2D projection error, 6D pose error, 5cm5degree error
        px_threshold = 5 # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works 
        eps          = 1e-5
        acc          = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
        acc3d10      = len(np.where(np.array(errs_3d) <= self.diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
        acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
        corner_acc   = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
        mean_err_2d  = np.mean(errs_2d)
        mean_corner_err_2d = np.mean(errs_corner2D)
        nts = float(testing_samples)

        # Print test statistics
        logging('\nResults of {}'.format(self.name))
        logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
        logging('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(self.diam * 0.1, acc3d10))
        logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
        logging("   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f" % (mean_err_2d, np.mean(errs_3d), mean_corner_err_2d))
        logging('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts) )
        pdb.set_trace()

        print("done with post process!")
        

    def truths_length(self, truths, max_num_gt=50):
        for i in range(max_num_gt):
            if truths[i][1] == 0:
                return i


    def run(self):
        rate = rospy.Rate(100)
        b_flag = True
        while not rospy.is_shutdown():
            try:
                rate.sleep()
            except: # this will happen if the clock goes backwards (i.e. rosbag loops)
                self.post_process_data()
                return


if __name__ == '__main__':
    try:
        np.set_printoptions(linewidth=160, suppress=True)  # make print statments easier to read
        program = ssp_rosbag()
        program.run()
    except:
        import traceback
        traceback.print_exc()
    print("done with program!")