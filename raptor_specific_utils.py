import numpy as np
import cv2
import os
import pdb

###########################################################
def draw_2d_proj_of_3D_bounding_box(img, corners2D_pr, corners2D_gt=None, epoch=None, batch_idx=None, detect_num=None, im_save_dir=None):
    """
    corners2D_gt/corners2D_pr is a 9x2 numpy array
    """
    open_cv_image = np.array(img)  # make it a numpy arr
    open_cv_image = np.moveaxis(255*np.squeeze(open_cv_image), 0, -1)[:, :, ::-1] # take out extra axis and go from (C, W, H) to (W, H, C). also 0-255 instead of 0 - 1
    open_cv_image = cv2.resize(open_cv_image, (640, 480))

    dot_radius = 2
    color_list = [(255, 0, 0),     # 0 blue: center
                  (0, 255, 0),     # 1 green: front lower right
                  (0, 0, 255),     # 2 red: front upper right
                  (255, 255, 0),   # 3 cyan: front lower left
                  (255, 0, 255),   # 4 magenta: front upper left
                  (0, 255, 255),   # 5 yellow: back lower right
                  (0, 0, 0),       # 6 black: back upper right
                  (255, 255, 255), # 7 white: back lower left
                  (125, 125, 125)] # 8 grey: back upper left
    if corners2D_gt is not None:
        for i, pnt in enumerate(corners2D_gt):
            open_cv_image = cv2.circle(open_cv_image, (pnt[0], pnt[1]), dot_radius, color_list[i], -1)  # -1 means filled in, else edge thickness

    inds_to_connect = [[1,2], [2, 4], [4, 3], [3, 1], # front face edges
                       [5,6], [6, 8], [8, 7], [7, 5], # back face edges
                       [1,5], [2, 6], [4, 8], [3, 7]] # side edges
    
    color_gt = (255,0,0)  # blue
    color_pr = (0,0,255)  # red
    linewidth = 1
    for inds in inds_to_connect:

        if corners2D_gt is not None:
            open_cv_image = cv2.line(open_cv_image, (corners2D_gt[inds[0],0], corners2D_gt[inds[0],1]), (corners2D_gt[inds[1],0], corners2D_gt[inds[1], 1]), color_gt, linewidth)
        open_cv_image = cv2.line(open_cv_image, (corners2D_pr[inds[0],0], corners2D_pr[inds[0],1]), (corners2D_pr[inds[1],0], corners2D_pr[inds[1], 1]), color_pr, linewidth)
    
    if im_save_dir is None:
        im_save_dir = "./backup/mslquad/test_output_images/"
    
    if batch_idx is None or batch_idx == 0:
        if not os.path.exists(im_save_dir):
            os.makedirs(im_save_dir)
    fn_str = "overlaid_img_"
    if epoch is not None:
        fn_str += 'epoch_{}'.format(epoch)
    if batch_idx is not None:
        if len(fn_str) > 0:
            fn_str += '_'
        fn_str += 'batch_{}'.format(batch_idx)
    if detect_num is not None:
        if len(fn_str) > 0:
            fn_str += '_'
        fn_str += 'detect_num_{}'.format(detect_num)

    cv2.imwrite(im_save_dir + "/" + fn_str + ".jpg", open_cv_image)

    return open_cv_image
###########################################################