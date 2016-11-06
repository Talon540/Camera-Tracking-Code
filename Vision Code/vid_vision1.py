#PLEASE NOTE, A LOT OF THIS CODE IS DEPRECATED. WE ONLY USED THE ANGULAR ALIGNMENT FUNCTION
from networktables import *

import numpy as np
import argparse
import imutils
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
import time as t
import cv2
#We used OpenCV 3.0.0

NetworkTable.setIPAddress("10.5.40.2")
NetworkTable.setClientMode()
print NetworkTable.port
sd = NetworkTable.getTable("cameratrack")
#Retrieves NetworkTable from RoboRIO, please change the IP address to that which your RIO is using
#Change the table name to that which you are using

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 64
rawCapture = PiRGBArray(camera, size=(320, 240))
#Initializes PiCamera

def onmouse(k,x,y,s,p):
        global hsv
        if k==1:   
                print(hsv[y,x])
        #If a static image is loaded and a point is clicked with Left Mouse,
        #displays the pixel coordinates of that point
def angle_cos(p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        squares = []
        for gray in cv2.split(img):
                for thrs in range(0, 255, 26):
                        if thrs == 0:
                                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                                bin = cv2.dilate(bin, None)
                        else:
                                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                        bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                                cnt_len = cv2.arcLength(cnt, True)
                                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                                        cnt = cnt.reshape(-1, 2)
                                        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                                        if max_cos < 0.1:
                                                squares.append(cnt)
        return squares


        shapeMask = cv2.inRange(img, lower, upper)
        retval, bin = cv2.threshold(shapeMask, 0, 255, cv2.THRESH_BINARY)
        (cnts, _) = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("Found", len(cnts))
        return cnts
#Finds square/rectangular shapes in the image (i.e: the shape of the goal)
def findcntsimple (img):
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127,255,0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
def distance_to_camera(knownWidth, focalLength, perWidth):
        #Compute and return the distance from the maker to the camera
        if perWidth>0:
                return -1.2172*perWidth +238.35#1.0*(knownWidth * focalLength) / perWidth
        else:
                return -1
#The Distance Calculation is inaccurate, since it depends on a straight 90 degree
#view of the goal, which is impossible during the match.
def get_midpoint(cnt):
        M = cv2.moments(cnt)
        cx = 1
        cy = 1
        #print(M)
        return cx, cy
#Finds middle point of the goal
t.sleep(0.1)
j = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True,): 
        #image =  cv2.imread('/home/pi/Desktop/2016vision/output187.jpg')
        image = frame.array
        #Determines what images to use for tracking, you can choose either a static
        #or an array of video frames
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 20,100])
        upper_green = np.array([110, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(image,image,mask=mask)
        j = j + 1
        #Initializes the HSV Filter, you may tweak the ranges as needed
        
        cnts = findcntsimple(res)
        #print(len(cnts))
        cv2.drawContours(image, cnts, -1, (0,255,0), 3)

        if cnts:
                c = max(cnts,key = cv2.contourArea)
                #goal is 1'8" wide, 1'2" tall target, goal is 2" inside
                KNOWN_DISTANCE = 42
                KNOWN_WIDTH = 20
                area = cv2.minAreaRect(c)
                
                x,y,w,h = cv2.boundingRect(c)
                #print(w)
                CALCULATED_FOCAL_LENGTH = 164.0*KNOWN_DISTANCE/KNOWN_WIDTH
                distance = distance_to_camera(KNOWN_WIDTH, CALCULATED_FOCAL_LENGTH, w)
                cv2.putText(image, "%.2fft" % (distance/12), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0),3)
                
                cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
                #draw cross hairs
                cx = x+int(w/2)
                cy = y+int(h/2)
                cv2.line(image, (cx-10,cy),(cx+10,cy) , (0,0,255))
                cv2.line(image, (cx,cy-10),(cx,cy+10) , (0,0,255))

                #calc dx
                dx = int(image.shape[1]/2) - cx
                #This distance is not accurate, so do not use it for alignment. Instead use an Ultrasonic sensor from the RIO's side
                if(int(image.shape[1]/2)>(x+int(w/2)-10)) and (int(image.shape[1]/2)<x+int(w/2) + 10):
                       print("Aligned with target, sir.",dx)
                       align = 1
                       print cx
                       print cy
                else:
                       print("We are not aligned, sir.", dx)
                       align = 0
                try:
                        #sd.putNumber("distance", distance)
                        #sd.putNumber('centerX', dx)
                        sd.putNumber('align', align)
                        print(sd.getNumber("align"))
                        print("Sending" + str(align))
                except KeyError:
                       print('alignment: N/A')
                #Sends a simple integer to NetworkTables. If aligned to the goal,
                #the Pi will send 1; if not aligned, it will send 0.
        #cv2.imshow('orig',image)
        #cv2.imshow("mask",mask)
        #cv2.imshow('result',res)
         
        #cv2.imshow('img_th',img_th)

        #cv2.namedWindow("hsv")
        cv2.setMouseCallback("orig",onmouse);
        #cv2.imshow('hsv',hsv)
        #You can uncomment the imshow functions if you are connected to a monitor
        #and wish to examine the output. Comment it when running headless during
        #a match
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        rawCapture.truncate(0)
        #Resets the current video frame to analyze the next frame

vid.release()
cv2.destroyAllWindows()
