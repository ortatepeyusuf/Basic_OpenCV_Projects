import cv2
import numpy as np

#you can change source paths
cap=cv2.VideoCapture("pixel_processing/images/car.mp4")

ret,first_frame=cap.read()

first_gray_frame=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
first_gray_frame=cv2.resize(first_gray_frame,(640,480))




while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,480))

    if frame is None:
        break
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    


    diff=cv2.absdiff(frame,first_gray_frame)

    cv2.imshow("frame",frame)
    cv2.imshow("diff",diff)

    first_gray_frame=frame
    


    if cv2.waitKey(3) & 0xFF==ord("q"):
        break




cap.release()
cv2.destroyAllWindows()
