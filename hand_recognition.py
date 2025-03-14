import cv2
import numpy as np

cap=cv2.VideoCapture(0)

lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

kernel=np.ones((5,5),dtype=np.uint8)

while True:
    ret,frame=cap.read()
    roi=frame[50:500,50:500]
    frame=cv2.flip(frame,1)

    cv2.putText(frame,"Put your hand here",(75,75),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

    cv2.rectangle(frame,(50,50),(500,500),(0,255,0),1)
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_skin,upper_skin)
    

    mask=cv2.dilate(mask,kernel,iterations=4)
    mask=cv2.GaussianBlur(mask,(5,5),0)

    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_contour=max(contours,key=cv2.contourArea)
    
    (x,y,w,h)=cv2.boundingRect(max_contour)

    cx=x+w/2

    if cx<200:
        cv2.putText(frame,"<--",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    if cx>200:
        cv2.putText(frame,"-->",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)


    cv2.imshow("LR",frame)

    if cv2.waitKey(1) ==27:
        break

cap.release()
cv2.destroyAllWindows()










        