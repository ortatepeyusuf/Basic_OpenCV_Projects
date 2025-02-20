import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners=cv2.goodFeaturesToTrack(gray_frame,200,0.01,50)
    corners=np.int0(corners)

    for corner in corners:
        x,y=corner.ravel()
        cv2.circle(frame,(x,y),4,(0,0,255),-1)

    cv2.imshow("Corner Detection",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


