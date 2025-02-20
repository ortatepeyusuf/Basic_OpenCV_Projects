import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edge_detected_frame=cv2.Canny(gray_frame,50,255)

    cv2.imshow("Original frame",frame)
    cv2.imshow("Edge Detected Binary Frame",edge_detected_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()




