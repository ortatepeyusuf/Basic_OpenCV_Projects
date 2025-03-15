import cv2
import numpy as np
from collections import deque


cap=cv2.VideoCapture(0)


kernel=np.ones((5,5),dtype=np.uint8)

lower_blue=np.array([100,60,60])
upper_blue=np.array([140,255,255])

blue_points=[deque(maxlen=512)]
green_points=[deque(maxlen=512)]
red_points=[deque(maxlen=512)]

colors=[(255,0,0),(0,255,0),(0,0,255)]
color_index=0



blue_index=0
green_index=0
red_index=0

window=np.ones((512,512,3))

cv2.rectangle(window,(40,1),(140,51),(0,0,0),2)
cv2.rectangle(window,(140,1),(240,51),colors[0],-1)
cv2.rectangle(window,(240,1),(340,51),colors[1],-1)
cv2.rectangle(window,(340,1),(440,51),colors[2],-1)


font=cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(window,"CLEAR ALL",(40,10),font,0.5,(0,0,0),2)
cv2.putText(window,"Blue",(140,10),font,0.5,(255,255,255),2)
cv2.putText(window,"Green",(240,10),font,0.5,(255,255,255),2)
cv2.putText(window,"Red",(340,10),font,0.5,(255,255,255),2)

cv2.namedWindow("Paint")


while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    
    cv2.rectangle(frame,(40,10),(140,51),(0,0,0),2)
    cv2.rectangle(frame,(140,10),(240,51),colors[0],-1)
    cv2.rectangle(frame,(240,10),(340,51),colors[1],-1)
    cv2.rectangle(frame,(340,10),(440,51),colors[2],-1)


    font=cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,"CLEAR ALL",(40,10),font,0.5,(0,0,0),2)
    cv2.putText(frame,"Blue",(140,10),font,0.5,(255,255,255),2)
    cv2.putText(frame,"Green",(240,10),font,0.5,(255,255,255),2)
    cv2.putText(frame,"Red",(340,10),font,0.5,(255,255,255),2)
    
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    mask=cv2.erode(mask,kernel,iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.dilate(mask,kernel,iterations=1)

    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    center=None

    if len(contours)>0:
        max_contour=sorted(contours,key=cv2.contourArea,reverse=True)[0]
        (x,y),radius=cv2.minEnclosingCircle(max_contour)
        cv2.circle(frame,(int(x),int(y)),int(radius),(255,0,255),3)

        M=cv2.moments(max_contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


        if center[1]<65:
            if center[0]<140 and center[0]>40:
                window = np.ones((512, 512, 3)) * 255 
                
                blue_points = [deque(maxlen=512)]
                green_points = [deque(maxlen=512)]
                red_points = [deque(maxlen=512)]
                blue_index = green_index = red_index = 0
            elif center[0]<240 and center[0]>140:
                color_index=0
            elif center[0]<340 and center[0]>240:
                color_index=1
            elif center[0]<440 and center[0]>340:
                color_index=2
        else:
            if color_index==0:
                    blue_points[blue_index].appendleft(center)
            elif color_index==1:
                green_points[green_index].appendleft(center)
            elif color_index==2:
                red_points[red_index].appendleft(center)
    else:
        blue_points.append(deque(maxlen=512))
        blue_index+=1
        green_points.append(deque(maxlen=512))
        green_index+=1
        red_points.append(deque(maxlen=512))
        red_index+=1
    points=[blue_points,green_points,red_points]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame,points[i][j][k-1],points[i][j][k],colors[i],1)
                cv2.line(window,points[i][j][k-1],points[i][j][k],colors[i],1)

    cv2.imshow("Paint",window)
    cv2.imshow("Frame",frame)
 
    if cv2.waitKey(10) ==27:
        break


cap.release()
cv2.destroyAllWindows()


        













