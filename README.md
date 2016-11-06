# Camera-Tracking-Code
Vision Tracking for 2016 FIRST Stronghold

We used a Raspberry Pi 2 & PiCamera to serve as a vision coprocessor, as we felt OpenCV would stress the RoboRIO.

## To Install OpenCV 3.0.0

The Operating System on the Pi was Raspbian Jessie

Consult the following guide for downloading OpenCV: http://www.pyimagesearch.com/2015/10/26/how-to-install-opencv-3-on-raspbian-jessie/

IMPORTANT: Be sure to have at least 8GB on the micro-SD card, OpenCV is very large! We recommend 16 GB as we nearly ran out of space on an 8GB micro-SD.

## Dependencies Needed (Excluding OpenCV)

1. Numpy

2. Imutils

3. picamera

4. time

5. networktables*

* The networktables library was actually PyNetworkTables, which you can refer to the documentation from here: http://pynetworktables.readthedocs.io/en/latest/

## Notes

1. A large portion of this code is deprecated since the end of last year's season, and some functions will not work with acceptable accuracy

  1a. The distance calculation function is not accurate, as this depends on having a straight 90 degree view of the goal, which is impossible during a match
  
2. Feel free to edit or modify this code as you please, just ensure to cite this source code from Team 540
