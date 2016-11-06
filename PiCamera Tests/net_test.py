from networktables import *
import time as t

NetworkTable.setClientMode


ct = NetworkTable.getTable("cameratrack")
center = 1

while True:
    ct.putNumber('x', center)
    t.sleep(1)
