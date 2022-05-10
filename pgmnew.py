
import cv2
from skimage.transform import resize
from skimage import util
from keras.models import load_model
import numpy as np

model = load_model('model.h5')
cbook = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O'
,'P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
# set up our ROI for grabbing the hand.
roi_top = 40
roi_bottom = 300
roi_right = 300
roi_left = 600
# select camera
cam = cv2.VideoCapture(0)

# Intialize a frame count
num_frames = 0

text = ''
cnt = 0
prev_ch =''
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
    if num_frames < 10:
        bg = roi
        continue
    
    val = cv2.absdiff(bg.astype("uint8"), roi)
    val = val.sum()
    # print(val)
    
    blackboard = np.zeros((100, 640, 3), dtype=np.uint8)
    # Grab the ROI from the frame
    
    roi = util.img_as_ubyte(resize(roi,(200,200)))
    roi = cv2.flip(roi, 1)
    roi = roi/255
    # roi = roi[:,:,0]
    # roi = cv2.GaussianBlur(roi, (11,11), 0)
    # roi= cv2.medianBlur(roi, 15)
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 2)
    cv2.putText(frame, "Place your hand inside the box", (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
    
    
    if val > 3025543:
        img = roi.copy()
        img = np.expand_dims(img,axis=0)
        # img = np.expand_dims(img,axis=3)
        out = model.predict(img)
        
        if out[0].max()>0.9:
            print(out[0].max())
            label = np.argmax(out[0])
            ch = cbook[label]
            print(cnt,prev_ch,ch,sep='--')
            
            if ch == prev_ch:
                cnt+=1
            else:
                prev_ch = ch
                cnt = 0
            if cnt>15:
                cnt = 0
                # print(ch)
                if ch == 'space':
                    ch = ' '
                elif ch =='nothing':
                    ch = ''
                elif ch =='del':
                    text =text[:-1]
                    ch =''
                text = text+ch
                print(text)
    cv2.putText(blackboard,text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)            
    cv2.imshow("Hand Gestures", np.vstack((frame, blackboard)))    
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()

