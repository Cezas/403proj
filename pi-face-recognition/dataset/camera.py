from picamera import PiCamera
import time
import sys


localtime = time.asctime(time.localtime(time.time()))
if len(sys.argv)==1:
	str =  localtime+'.jpg'
else:
	str =sys.argv[1]+'.jpg'
camera = PiCamera()
camera.start_preview()
time.sleep(5)
camera.capture(str)
camera.stop_preview()
