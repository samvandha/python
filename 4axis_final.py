#### 4 axis robotic arm simulation####

import numpy as np
import cv2 as cv
import time
import os
from sympy import symbols
os.chdir(r'/home/samvandha/Desktop')

arm1 = 100
center = [400,300]
print(center[1])
arm2 = 80
zangle = 0
notgrab = 1
reverse = 0
points= [[410,130,0], [500,250,1], [580,290,0], [470,250,1], [470,127,1], [430,290,0], [500,300,0]]
#points= [[410,130,0], [500,250,1]]
#points = [[550,300]]
#points= [[410,130,0]]

def calculate_angles(x1,y1):
    ###Find the point where the circle from arm 2 cuts the circle from arm1
    d = ((x1-center[0])**2 + (y1-center[1])**2)**0.5
    if x1 - center[0] != 0:
        angle_bw = np.arctan((-y1+center[1])/(x1-center[0]))
    else:
        angle_bw = np.pi/2
    a= (arm1**2-arm2**2+d**2)/(2*d)
    b = d - a
    h = (arm1**2-a**2)**0.5
    if a != 0:
        theta1 = np.arctan(h/a)
    else:
        theta1 = 0
    angle = angle_bw + theta1
    x1 = center[0] + arm1*np.cos(angle)
    y1 = center[1] - arm1*np.sin(angle)
    return(int(x1),int(y1))

def move(current_angle1, current_angle2, angle_change1, angle_change2):
    x1,y1 = int(center[0]+arm1*np.cos(current_angle1)), int(center[1]-arm1*np.sin(current_angle1))
    x2, y2 = int(x1+arm2*np.cos(current_angle2)), int(y1-arm2*np.sin(current_angle2))
    current_angle1 += angle_change1
    current_angle2 += angle_change2
    return(x1,y1,x2,y2,current_angle1,current_angle2)

def calculate(x1,y1,x2,y2):
    x_new, y_new = calculate_angles(points[i][0], points[i][1])
    old_point = (x1,y1)
##    print("X_new:", x_new, "Y_new:", y_new)
##    print("X1:", x1, "Y1", y1)
    if (x_new-400) != 0:
        new_angle1 = np.arctan2((-y_new+300),(x_new-400))
    else:
        new_angle1 = np.pi/2
        
    if (x1-400) != 0:
        old_angle1 = np.arctan2((-y1+300),(x1-400))
    else:
        old_angle1 = np.pi/2
        
    angle_change1 = ((old_angle1-new_angle1)**2)**0.5/10
    print("Angle to change:", angle_change1)
    if (x2-x1) != 0:
        old_angle2 = np.arctan2((-y2+y1),(x2-x1))
    else:
        old_angle2 = np.pi/2
    if (points[i][0]-x_new):
        new_angle2 = np.arctan2((-points[i][1]+y_new),(points[i][0]-x_new))
    else:
        new_angle2 = np.pi/2
    angle_change2 = ((old_angle2-new_angle2)**2)**0.5/10
    if old_angle1 > new_angle1:
        angle_change1 = -angle_change1
    if old_angle2 > new_angle2:
        angle_change2 = -angle_change2
    current_angle1 = old_angle1
    current_angle2 = old_angle2
##    print("Old angle1: ", 180*old_angle1/np.pi)
##    print("New angle1: ", 180*new_angle1/np.pi)
##    print("Old angle2: ", 180*old_angle2/np.pi)
##    print("New angle2: ", 180*new_angle2/np.pi)
    return(angle_change1, angle_change2,current_angle1,current_angle2)


i = 0
j = 0
k = 0
flag_change = 0
while(1):
    img = cv.imread('blank.jpg', 1)
    img = cv.resize(img, (800,600), interpolation = cv.INTER_AREA)  
    #x1, y1 = 400+int(arm1*np.cos(angle1)),300-int(arm1*np.cos(cangle1))
    if flag_change == 0: 
        x1,y1 = calculate_angles(points[i][0], points[i][1])
        x2, y2 = points[i][0], points[i][1]
        if (x1-400) != 0:
            old_angle1 = np.arctan2((-y1+300),(x1-400))
        else:
            old_angle1 = np.pi/2
        if (x2-x1) != 0:
            old_angle2 = np.arctan2((-y2+y1),(x2-x1))
        else:
            old_angle2 = np.pi/2
        current_angle1 = old_angle1
        current_angle2 = old_angle2
    else:
        x1,y1,x2,y2,current_angle1,current_angle2 = move(current_angle1, current_angle2, angle_change1, angle_change2)
        
    if k == 10:
        k = 0
        if flag_change == 0:
            print("Flag changed to 1")
            flag_change = 1
            i += 1
            if i > len(points)-1:
                i = 0
            angle_change1, angle_change2,current_angle1,current_angle2 = calculate(x1,y1,x2,y2)
            print("Calculated movement")
            #print("Current angle1: ", 180*current_angle1/np.pi)
        else:
            flag_change = 0
    k += 1
    
    img = cv.line(img, (400,300), (800,300), (0,255,0), 2)
    img = cv.line(img, (400, 300), (400,0), (255,0,0), 2)
    img = cv.line(img, (400,300), (int(400-400*np.cos(np.pi/4)),600), (0,0,255), 2)
    cv.circle(img, (400,300), 180, (0,200,200), 2)
    cv.circle(img, (400,300), 100, (0,200,200), 2)
    cv.circle(img, (x1,y1), 80, (0, 100, 200), 2)
    cv.ellipse(img, (400,300), (30,4), 0, 0, 360, (0,0,255), thickness=-1, lineType=1, shift=0)
    cv.circle(img, (400,300), 8, (255,0,0), -1)
    #First arm
    img = cv.line(img, (400,300), (x1,y1), (200,0,200), 5)
    cv.circle(img, (x1,y1), 8, (255,0,0), -1)
    #Second arm
    img = cv.line(img,(x1,y1), (x2,y2), (200,0,200), 5)
    cv.circle(img, (x2,y2), 8, (255,0,0), -1)
    #third
    img = cv.line(img, (x2,y2), (x2-5*notgrab,y2+20), (0,0,0), 2)
    img = cv.line(img, (x2,y2), (x2+5*notgrab,y2+20), (0,0,0), 2)
    cv.putText(img, "X Axis", (650, 290), 2, 1.0, (0,255,0))
    cv.putText(img, "Y Axis", (410, 30), 2, 1.0, (255,0,0))
    cv.putText(img, "Z Axis", (180, 570), 2, 1.0, (0,0,255))
    cv.putText(img, "Angle1: "+ str(180*current_angle1/np.pi), (450, 500), 2, 1.0, (0,0,255))
    cv.putText(img, "Angle2: "+ str(180*current_angle2/np.pi), (450, 530), 2, 1.0, (0,0,255))
    
    cv.imshow("Image", img)
    #print("I", i)

    time.sleep(0.1)

    k1 = cv.waitKey(1) & 0xFF
    if k1 == 27:
        break

cv.destroyAllWindows()
