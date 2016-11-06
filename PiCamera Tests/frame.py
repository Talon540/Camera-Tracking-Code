import numpy as np
import argparse
import imutils
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
import time as t
import cv2

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera)

camera.capture(rawCapture, format="bgr")
image = rawCapture.array

t.sleep(1)

cv2.imshow("Image", image)
cv2.waitKey(0)
