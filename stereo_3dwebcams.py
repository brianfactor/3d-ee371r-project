#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )

Modified significantly by Brian Morgan, 16 Apr 2013
'''

import numpy as np
import cv2
import sys

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

CAM_L = 1
CAM_R = 2

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


if __name__ == '__main__':
    #print 'loading images...'
    #imgL = cv2.pyrDown( cv2.imread('img/left100.ppm') )  # downscale images for faster processing
    #imgR = cv2.pyrDown( cv2.imread('img/right100.ppm') )
    
    if len(sys.argv) == 1:
        # image preview (allows it to adjust for brightness
        devL = cv2.VideoCapture(CAM_L)
        while cv2.waitKey(50) == -1:   # key is pressed on windows
            _,imgL = devL.read()
            cv2.imshow('left', imgL)            
        devL.release()
    
        devR = cv2.VideoCapture(CAM_R)
        while cv2.waitKey(50) == -1:
            s,imgR = devR.read()
            cv2.imshow('right', imgR)
        devR.release()
    
        print 'capturing images from webcams'
       
        devL = cv2.VideoCapture(CAM_L)
        _,imgL = devL.read()
        devL.release()
        devR = cv2.VideoCapture(CAM_R)
        _,imgR = devR.read()
        devR.release()
        cv2.imshow('left',imgL)
        cv2.imwrite('l.jpg',imgL)
        cv2.imshow('right',imgR)
        cv2.imwrite('r.jpg',imgR)
        imgRGB = imgL

        imgL = cv2.imread('l.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)  # downscale img
        imgR = cv2.imread('r.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
        
    else:
        imgL = cv2.imread(sys.argv[1])
        imgR = cv2.imread(sys.argv[2])
        imgRGB = imgL
    
    # load camera calibration
#    Q = np.float32(cv2.cv.Load('cal/Q.xml', cv2.cv.CreateMemStorage()))
#    print 'Q ='
#    print Q
    mx1 = np.float32(cv2.cv.Load('cal/mx1.xml', cv2.cv.CreateMemStorage()))
    my1 = np.float32(cv2.cv.Load('cal/my1.xml', cv2.cv.CreateMemStorage()))
    mx2 = np.float32(cv2.cv.Load('cal/mx2.xml', cv2.cv.CreateMemStorage()))
    my2 = np.float32(cv2.cv.Load('cal/my2.xml', cv2.cv.CreateMemStorage()))

    # rectify for lense distortion
    imgL = cv2.remap(imgL, mx1, my1, cv2.INTER_LINEAR)
    imgR = cv2.remap(imgR, mx2, my2, cv2.INTER_LINEAR)
    cv2.imshow('left rectified', imgL)
    cv2.imshow('right rectified', imgR)
    cv2.waitKey()

    # disparity range is tuned for 'aloe' image pair
    window_size = 10 # 3 for outside
    min_disp = 16
    num_disp = 112-min_disp
    # Using Block Matching algorithm (fast)
    stereo = cv2.StereoSGBM(minDisparity = min_disp,
        numDisparities = num_disp,
        SADWindowSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 175,
        speckleRange = 64,
        disp12MaxDiff = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        fullDP = False
    )

    print 'computing disparity...'
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    
    #disp = cv2.bilateralFilter(disp,10,20,20)
    #cv2.imshow('disparity filtered', (disp-min_disp)/num_disp)
    
    # filter the disparity map
    #stereo.filterSpeckles(disp, 0, 50, 4)
    #cv2.imshow('disparity filtered', (disp-min_disp)/num_disp)
    
    # Using Graph Cutting (more accurate)
    # http://opencv.willowgarage.com/documentation/python/calib3d_camera_calibration_and_3d_reconstruction.html
    #(r, c) = (imgL.rows, imgL.cols)
    #dispL = cv.CreateMat(r, c, cv.CV_16S)
    #dispR = cv.CreateMat(r, c, cv.CV_16S)
    #state = cv.CreateStereoGCState(16, 2)
    #cv.FindStereoCorrespondenceGC(imgL, imgR, dispL, dispR, state, 0)
    #disp = cv.CreateMat(r, c, cv.CV_8U)
    #ConvertScale(dispL, disp, -16)

    print 'generating 3d point cloud...',
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    print Q

    points = cv2.reprojectImageTo3D(disp, Q)
    imgRGB = cv2.remap(imgRGB, mx1, my1, cv2.INTER_LINEAR)
    colors = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    
    cv2.imwrite('disp.jpg', disp)
    cv2.imwrite('l_r.jpg',imgL)
    cv2.imwrite('r_r.jpg',imgR)
    
    print '%s saved' % 'out.ply'

    cv2.waitKey()
    cv2.destroyAllWindows()
    
