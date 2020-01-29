# IMPORTS
# system
import os
import pdb
# math
import numpy as np
import numpy.linalg as la
# Vision
import cv2
try:
    from cv_bridge import CvBridge, CvBridgeError
except:
    print("WARNING: ros not install so undistort_ros_image() function will be unavailable")

class camera:
    def __init__(self, K, dist_coefs, im_w, im_h, tf_cam_ego):
        """
        K: camera intrinsic matrix 
        tf_cam_ego: camera pose relative to the ego_quad (fixed)
        fov_horz/fov_vert: Angular field of view (IN RADIANS) for horizontal and vertical directions
        fov_lim_per_depth: how the boundary of the fov (width, heigh) changes per depth
        """
        self.K = K
        self.dist_coefs = dist_coefs
        self.im_w = im_w
        self.im_h = im_h
        if self.dist_coefs is None:
            self.new_camera_matrix = K
        else:
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (im_w, im_h), 0, (im_w, im_h))

        self.K_inv = la.inv(self.K)
        self.new_camera_matrix_inv = la.inv(self.new_camera_matrix)
        self.tf_cam_ego = tf_cam_ego

        (self.fov_horz, self.fov_vert), self.fov_lim_per_depth = self.calc_fov()

        try:
            self.bridge = CvBridge()
        except:
            pass

    def pnt3d_to_pix(self, pnt_c):
        """
        input: assumes pnt in camera frame
        output: [row, col] i.e. the projection of xyz onto camera plane
        """
        rc = np.matmul(self.new_camera_matrix, np.reshape(pnt_c[0:3], 3, 1))
        rc = np.array([rc[1], rc[0]]) / rc[2]
        return rc

    
    def b_is_pnt_in_fov(self, pnt_c, buffer=0):
        """ 
        - Use similar triangles to see if point (in camera frame!) is beyond limit of fov 
        - buffer: an optional buffer region where if you are inside the fov by less than 
            this the function returns false
        """
        if pnt_c[2] <= 0:
            raise RuntimeError("Point is at or behind camera!")
            return False
        fov_lims = pnt_c[2] * self.fov_lim_per_depth - buffer
        return np.all( np.abs(pnt_c[0:2]) < fov_lims )
        

    def calc_fov(self):
        """
        - Find top, left point 1 meter along z axis in cam frame. the x and y values are 
        half the width and height. Note: [x_tl, y_tl, 1 (= z_tl)] = inv(K) @ [0, 0, 1], 
        which is just the first tow rows of the third col of inv(K).
        - With these x and y, just use geometry (knowing z dist is 1) to get the angle 
        spanning the x and y axis respectively.
        - keeping the width and heigh of the point at 1m depth is useful for similar triangles
        """
        fov_lim_per_depth = -la.inv( self.new_camera_matrix )[0:2, 2] 
        return 2 * np.arctan( fov_lim_per_depth ), fov_lim_per_depth


    def undistort_ros_image(self, img_msg):
        """
        input: image as a ros message
        output: undistorted image as opencv format
        """
        image_cv = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        image_cv = cv2.undistort(image_cv, self.K, self.dist_coefs, None, self.new_camera_matrix)
        return image_cv
