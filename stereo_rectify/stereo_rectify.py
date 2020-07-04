#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob


# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
# imgpoints1 for left, imgpoints2 for right
imgpoints1 = []
imgpoints2 = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= 20.64  # mm : square size of chess board
prev_img_shape = None

# Extracting path of individual image stored in a given directory
lefts = glob.glob('../Project_Stereo_left/left/*.jpg')
rights = glob.glob('../Project_Stereo_right/right/*.jpg')
for lname, rname  in zip(lefts, rights):
    limg = cv2.imread(lname)
    rimg = cv2.imread(rname)

    lgray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    rgray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    Lret, Lcorners = cv2.findChessboardCorners(lgray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    Rret, Rcorners = cv2.findChessboardCorners(rgray, CHECKERBOARD,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if Lret == True and Rret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        Lcorners2 = cv2.cornerSubPix(lgray, Lcorners, (11, 11), (-1, -1), criteria)
        Rcorners2 = cv2.cornerSubPix(rgray, Rcorners, (11, 11), (-1, -1), criteria)

        imgpoints1.append(Lcorners2)
        imgpoints2.append(Rcorners2)

        # Draw and display the corners
        '''
        img1 = cv2.drawChessboardCorners(limg, CHECKERBOARD, Lcorners2, Lret)
        img2 = cv2.drawChessboardCorners(rimg, CHECKERBOARD, Rcorners2, Rret)

    cv2.imshow(lname, img1)
    cv2.imshow(rname, img2)
    cv2.waitKey(0)
    

cv2.destroyAllWindows()

lh, lw = img1.shape[:2]
rh, rw = img2.shape[:2]
'''

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
Lret, Lmtx, Ldist, Lrvecs, Ltvecs = cv2.calibrateCamera(objpoints, imgpoints1, lgray.shape[::-1], None, None)
Rret, Rmtx, Rdist, Rrvecs, Rtvecs = cv2.calibrateCamera(objpoints, imgpoints2, rgray.shape[::-1], None, None)

ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpoints1,imgpoints2,Lmtx,Ldist,Rmtx,Rdist,lgray.shape[::-1])

R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(mtx1,dist1,mtx2,dist2,lgray.shape[::-1],R,T,alpha=1)

map11, map12 = cv2.initUndistortRectifyMap(mtx1,dist1,R1,P1,lgray.shape[::-1],cv2.CV_16SC2)
map21, map22 = cv2.initUndistortRectifyMap(mtx2,dist2,R2,P2,rgray.shape[::-1],cv2.CV_16SC2)


l_img_test = cv2.imread(lefts[0])
r_img_test = cv2.imread(rights[0])

l_dst_test = cv2.remap(l_img_test,map11,map12,cv2.INTER_LINEAR)
r_dst_test = cv2.remap(r_img_test,map21,map22,cv2.INTER_LINEAR)

res_test = cv2.hconcat([l_dst_test,r_dst_test])

h,w = l_dst_test.shape[:2]
delta=20
new_h = int(h/delta)
for id in range(new_h):
    cv2.line(res_test,(0,id*delta),(2*w,id*delta),(255,0,0))

cv2.rectangle(res_test,roi1[:2],roi1[2:4],(0,0,255))
cv2.rectangle(res_test,(roi2[0]+w,roi2[1]),(roi2[2]+w,roi2[3]),(0,0,255))

cv2.imshow('res',res_test)
cv2.waitKey(0)
