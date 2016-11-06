from networktables import *
import time as t

NetworkTable.setIPAddress("10.5.40.2")
NetworkTable.setClientMode
print NetworkTable.port
ct = NetworkTable.getTable("cameratrack")

center = 1

while True:
    ct.putNumber('x', center)
    t.sleep(1)
