import cv2
import numpy as np




#you can change source paths

cap=cv2.VideoCapture("pixel_processing\\images\\eye_motion.mp4")


while True:
    ret,frame=cap.read()
    roi=frame[80:210,230:450]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

    _,threshold=cv2.threshold(gray,5,255,cv2.THRESH_BINARY_INV)

    contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)

    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.line(roi,(x,int(y+h/2)),(x+w,int(y+h/2)),(0,0,255),1)
        cv2.line(roi,(int(x+w/2),y),(int(x+w/2),y+h),(0,0,255),1)
        break
    
    frame[80:210,230:450]=roi

    
    cv2.imshow("frame",frame)


    if cv2.waitKey(30) & 0xFF==ord("q"):
        break


cap.release()
cv2.destroyAllWindows()



