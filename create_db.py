import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.util import img_as_ubyte
import os

# set up our ROI for grabbing the hand.
roi_top = 40
roi_bottom = 300
roi_right = 300
roi_left = 600
# select camera
cam = cv2.VideoCapture(0)

# Intialize a frame count
num_frames = 0
cnt =1;
# start
while True:
    # get the current frame
    ret, frame = cam.read()
    if not(ret):
      continue
    frame = cv2.flip(frame, 1)
    
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    num_frames +=1
    

    # Grab the ROI from the frame
    
    roi = img_as_ubyte(resize(roi,(100,100)))
    roi = cv2.flip(roi, 1)
    # roi = roi[:,:,0]
    # roi = cv2.GaussianBlur(roi, (7, 7), 0)
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 2)
    cv2.putText(frame, "Place your hand in side the box", (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
    cv2.putText(frame,str(cnt), (400, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)
    
    if num_frames%30==0:
        filename = str(cnt)+'.png'
        filepath = os.path.join('dataset',filename)
        plt.imsave(filepath,roi)
        
        cnt+=1
    cv2.imshow("Hand Gestures", frame)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()

