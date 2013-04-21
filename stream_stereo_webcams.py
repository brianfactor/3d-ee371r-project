#!/usr/bin/env python

# Big problem: this capture routine does no brightness adjustments. None. (for demo)

import cv2
#import cv
#import time

CAM_L = 1
CAM_R = 2

if __name__ == '__main__':
    print 'starting stereo webcam stream...'
    
    cv2.namedWindow('left',1)
    cv2.namedWindow('right',1)
    
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
    
    i = 0
    try:
        while True:
            # better way to do it (that doesn't work):
            #devL.open()
            #devR.open()
            #devL.grab()
            #devR.grab()
            #s,imgL = devL.retrieve()
            #devL.release()
            devL = cv2.VideoCapture(CAM_L)
            devL.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
            devL.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
            _,imgL = devL.read()
            devL.release()
            devR = cv2.VideoCapture(CAM_R)
            devR.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
            devR.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
            s,imgR = devR.read()
            devR.release()
            
            cv2.imshow('left', imgL)
            cv2.imshow('right', imgR)
            #cv2.cv.moveWindow('left', 0, 0)
            #cv2.cv.moveWindow('right', 512, 0)
            i = i + 1
            if cv2.waitKey(50) != -1:   # key is pressed on windows
                cv2.imwrite('img/left%d.ppm' % i, imgL)
                cv2.imwrite('img/right%d.ppm' % i, imgR)
                #break
    except KeyboardInterrupt:           # ^C is pressed in terminal
        devL.release()
        devR.release()
        cv2.destroyAllWindows()

    print 'ended by keyboard interrupt (^C)'

