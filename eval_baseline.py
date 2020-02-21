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
import numpy.linalg as la

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
from sensor_msgs.msg import Image as ROS_IMAGE
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from raptor_logger import *
from pose_metrics import *

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
        rb_name = rospy.get_param('~rb_name')
        self.ado_names = [rospy.get_param('~tracked_name')]

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

        self.ego_name     = data_options['name']
        gpus              = data_options['gpus'] 
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
        self.ego_pose_est_msg_buf = []
        self.ego_pose_est_time_msg_buf = []
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


        self.log_out_dir = '/mounted_folder/ssp_logs'
        # ssp_log_name    = self.log_out_dir + "/log_" + rb_name.split("_")[-1] + "_SSP.log"
        # param_log_name = self.log_out_dir + "/log_" + rb_name.split("_")[-1] + "_PARAM.log"
        # self.logger = raptor_logger(source="SSP", mode="write", ssp_fn=ssp_log_name, param_fn=param_log_name)
        base_path = self.log_out_dir + "/log_" + rb_name.split("_")[-1]
        self.rb_name = rb_name
        self.bb_3d_dict_all = {self.ado_names[0] : [box_length, box_width, box_height, self.diam]}
        self.logger = RaptorLogger(mode="write", names=self.ado_names, base_path=base_path, b_ssp=True)

        # Write params to log file ########################################################################################################
        param_data = {}
        if self.new_camera_matrix is not None:
            param_data['K'] = np.array([self.new_camera_matrix[0, 0], self.new_camera_matrix[1, 1], self.new_camera_matrix[0, 2], self.new_camera_matrix[1, 2]])
        else:
            param_data['K'] = np.array([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]])
        param_data['3d_bb_dims'] = np.array([box_length, box_width, box_height, self.diam])
        param_data['tf_cam_ego'] = np.reshape(copy(self.tf_cam_ego), (16,))
        # self.logger.write_data_to_log(log_data, mode='prms')
        self.logger.write_params(param_data)
        ###################################################################################################################################
        self.t0 = None
        # self.raptor_metrics = pose_metric_tracker(px_thresh=5, prct_thresh=10, trans_thresh=0.05, ang_thresh=5, name=self.name, diam=self.diam)
        self.raptor_metrics = PoseMetricTracker(px_thresh=5, prct_thresh=10, trans_thresh=0.05, ang_thresh=5, names=self.ado_names, bb_3d_dict=self.bb_3d_dict_all)
        
        rospy.Subscriber(self.ns + '/mavros/vision_pose/pose', PoseStamped, self.ego_pose_gt_cb, queue_size=10)  # optitrack pose
        rospy.Subscriber(self.ns + '/mavros/local_position/pose', PoseStamped, self.ego_pose_est_cb, queue_size=10)  # onboard ekf pose est
        rospy.Subscriber('/quad4' + '/mavros/vision_pose/pose', PoseStamped, self.ado_pose_gt_cb, queue_size=10)  # optitrack pose
        rospy.Subscriber(self.ns + '/camera/image_raw', ROS_IMAGE, self.image_cb, queue_size=1,buff_size=2**21)

        
    def ado_pose_gt_cb(self, msg):
        # if self.first_time is not None and self.first_time >= msg.header.stamp.to_sec():
        #     return
        self.ado_pose_gt_rosmsg = msg.pose
        pose_tm = msg.header.stamp.to_sec()
        self.ado_pose_msg_buf.append(msg)
        self.ado_pose_time_msg_buf.append(pose_tm)


    def ego_pose_gt_cb(self, msg):
        # if self.first_time is not None and self.first_time >= msg.header.stamp.to_sec():
        #     return
        self.ego_pose_gt_rosmsg = msg.pose
        pose_tm = msg.header.stamp.to_sec()
        self.ego_pose_msg_buf.append(msg)
        self.ego_pose_time_msg_buf.append(pose_tm)

    def ego_pose_est_cb(self, msg):
        # if self.first_time is not None and self.first_time >= msg.header.stamp.to_sec():
        #     return
        self.ego_pose_est_rosmsg = msg.pose
        pose_tm = msg.header.stamp.to_sec()
        self.ego_pose_est_msg_buf.append(msg)
        self.ego_pose_est_time_msg_buf.append(pose_tm)


    def image_cb(self, msg):
        """
        Maintains a buffer of images & times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        img_tm = msg.header.stamp.to_sec()
        if len(program.result_list) > 0 and img_tm <= self.result_list[-1][5]:
            return

        if self.t0 is None:
            self.t0 = img_tm

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

        tf_cam_ado_est = rotm_and_t_to_tf(R_pr, t_pr)

        if len(self.ado_pose_time_msg_buf) == 0 or len(self.ego_pose_time_msg_buf) == 0 or len(self.ego_pose_est_time_msg_buf) == 0:
            print("still waiting for other rosbag messages")
            return

        ado_msg, _ = find_closest_by_time(img_tm, self.ado_pose_time_msg_buf, message_list=self.ado_pose_msg_buf)
        ego_gt_msg, _ = find_closest_by_time(img_tm, self.ego_pose_time_msg_buf, message_list=self.ego_pose_msg_buf)
        ego_est_msg, _ = find_closest_by_time(img_tm, self.ego_pose_est_time_msg_buf, message_list=self.ego_pose_est_msg_buf)

        tf_w_ado_gt = pose_to_tf(ado_msg.pose)
        tf_w_ego_gt = pose_to_tf(ego_gt_msg.pose)
        tf_w_ego_est = pose_to_tf(ego_est_msg.pose)

        tf_w_cam_gt = tf_w_ego_gt @ invert_tf(self.tf_cam_ego)

        tf_w_ado_est = tf_w_cam_gt @ tf_cam_ado_est

        tf_w_ado_est[0:3, 0:3] = quat_to_rotm(remove_yaw(rotm_to_quat(tf_w_ado_est[0:3, 0:3])))  # remove yaw
        tf_w_ado_gt[0:3, 0:3] = quat_to_rotm(remove_yaw(rotm_to_quat(tf_w_ado_est[0:3, 0:3])))  # remove yaw
        
        quat_pr = rotm_to_quat(tf_w_ado_est[0:3, 0:3])
        state_pr = np.concatenate((tf_w_ado_est[0:3, 3], quat_pr))  # shape = (7,)

        img_to_save = copy(np.array(img.cpu()))
        # print("..................")
        # print("self.tf_cam_ego:\n{}".format(self.tf_cam_ego))
        # print("tf_cam_ado_est:\n{}".format(tf_cam_ado_est))
        # print("tf_cam_ado_gt = invert_tf(tf_w_cam_gt) @ tf_w_ado_gt:\n{}".format(invert_tf(tf_w_cam_gt) @ tf_w_ado_gt))

        self.result_list.append((state_pr, copy(tf_w_ado_est), copy(tf_w_ado_gt), copy(corners2D_pr), img_to_save, img_tm, time.time(), copy(R_pr), copy(t_pr), invert_tf(tf_w_cam_gt), copy(tf_w_ego_gt), copy(tf_w_ego_est)) )

        del img
        self.itr += 1
        if self.itr > 0 and self.itr % 50 == 0:
            print("Finished processing image #{}".format(self.itr))
            torch.cuda.empty_cache()


    def post_process_data(self):
        print("Post-processing data now ({} itrs)".format(len(self.result_list)))
        b_save_bb_imgs = True
        name = self.ado_names[0]
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
        log_data = {}
        for i, res in enumerate(self.result_list):

            # extract /  compute values for comparison
            state_pr, tf_w_ado_est, tf_w_ado_gt, corners2D_pr, img, img_tm, sys_time, R_cam_ado_pr, t_cam_ado_pr, tf_cam_w_gt, tf_w_ego_gt, tf_w_ego_est = res
            tf_cam_ado_gt = tf_cam_w_gt @ tf_w_ado_gt
            R_cam_ado_gt = tf_cam_ado_gt[0:3, 0:3]
            t_cam_ado_gt = tf_cam_ado_gt[0:3, 3].reshape(t_cam_ado_pr.shape)
            # if i == 0:
            #     pdb.set_trace()
            if i > 400 and self.rb_name == "rosbag_for_post_process_2019-12-18-02-10-28":
                print("STOPPING EARLY")
                break # quad crashes
            # if la.norm(tf_cam_ado_gt[0:3, 3] - t_cam_ado_pr) > 4:
            #     print("skipping (norm = {:.3f}".format(la.norm(tf_cam_ado_gt[0:3, 3] - t_cam_ado_pr)))
            #     continue
            # print(".")

            Rt_cam_ado_gt = np.concatenate((R_cam_ado_gt, t_cam_ado_gt), axis=1)
            Rt_cam_ado_pr = np.concatenate((R_cam_ado_pr, t_cam_ado_pr), axis=1)
            corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), self.vertices)), Rt_cam_ado_gt, self.new_camera_matrix).T
      
            if b_save_bb_imgs:
                draw_2d_proj_of_3D_bounding_box(img, corners2D_pr, corners2D_gt=corners2D_gt, epoch=None, batch_idx=None, detect_num=i, im_save_dir=bb_im_path)

            if self.raptor_metrics is not None:
                # self.raptor_metrics.update_all_metrics(vertices=self.vertices, R_gt=R_gt, t_gt=t_gt, R_pr=R_pr, t_pr=t_pr, K=self.new_camera_matrix)
                self.raptor_metrics.update_all_metrics(name=name, vertices=self.vertices, tf_w_cam=invert_tf(tf_cam_w_gt), R_cam_ado_gt=R_cam_ado_gt, t_cam_ado_gt=t_cam_ado_gt, R_cam_ado_pr=R_cam_ado_pr, t_cam_ado_pr=t_cam_ado_pr, K=self.new_camera_matrix)


            # Write data to log file #############################
            log_data['time'] = img_tm - self.t0
            log_data['state_est'] = tf_to_state_vec(tf_w_ado_est)
            log_data['state_gt'] = tf_to_state_vec(tf_w_ado_gt)
            log_data['ego_state_est'] = tf_to_state_vec(tf_w_ego_est)
            log_data['ego_state_gt'] = tf_to_state_vec(tf_w_ego_gt)
            corners3D_pr = (tf_w_ado_est @ self.vertices)[0:3,:]
            corners3D_gt = (tf_w_ado_gt @ self.vertices)[0:3,:]
            log_data['corners_3d_est'] = np.reshape(corners3D_pr, (corners3D_pr.size,))
            log_data['corners_3d_gt'] = np.reshape(corners3D_gt, (corners3D_gt.size,))
            log_data['proj_corners_est'] = np.reshape(self.raptor_metrics.proj_2d_pr[name].T, (self.raptor_metrics.proj_2d_pr[name].size,))
            log_data['proj_corners_gt'] = np.reshape(self.raptor_metrics.proj_2d_gt[name].T, (self.raptor_metrics.proj_2d_gt[name].size,))


            log_data['x_err'] = tf_w_ado_est[0, 3] - tf_w_ado_gt[0, 3]
            log_data['y_err'] = tf_w_ado_est[1, 3] - tf_w_ado_gt[1, 3]
            log_data['z_err'] = tf_w_ado_est[2, 3] - tf_w_ado_gt[2, 3]
            log_data['ang_err'] = calcAngularDistance(tf_w_ado_est[0:3, 0:3], tf_w_ado_gt[0:3, 0:3])
            log_data['pix_err'] = np.mean(la.norm(self.raptor_metrics.proj_2d_pr[name] - self.raptor_metrics.proj_2d_gt[name], axis=0))
            log_data['measurement_dist'] = la.norm(tf_w_ego_gt[0:3, 3] - tf_w_ado_gt[0:3, 3])

            self.logger.write_data_to_log(log_data, name, mode='ssp')
            self.logger.write_data_to_log(log_data, name, mode='ssperr')

            if np.any(np.isnan(corners3D_pr)) or np.any(np.isnan(corners3D_gt)) or np.any(np.isnan(self.raptor_metrics.proj_2d_pr[name])): #or la.norm(tf_cam_ado_gt[0:3, 3] - t_cam_ado_pr) > 10:
                print("ISSUE DETECTED!!")
                pdb.set_trace()
            ######################################################
        if self.raptor_metrics is not None:
            self.raptor_metrics.calc_final_metrics()
            self.raptor_metrics.print_final_metrics()

        self.logger.close_files()
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