# myapp.py

#from random import random
import os
from bokeh.layouts import column,row
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc, show
from cv2 import imread
from multiprocessing import Process

# create a plot and style its properties
p = figure(x_range=(0, 100), y_range=(0, 100))
p.image_url(url=['http://personal.psu.edu/rjy5060/othersentry.jpg'],x=0,y=100,w=100,h=100)

# create a callback that will add a number in a random location
def shutdown():
	#print("SHUTTING DOWN")
	f = open('target.txt','w')
	f.write('q')
	f.close()	
	p.image_url(url=['http://personal.psu.edu/rjy5060/othersentry.jpg'],x=0,y=100,w=100,h=100)
	
def face():
	#print("face")
	f = open('target.txt','w')
	f.write('h')
	f.close()   
	p.image_url(url=['http://personal.psu.edu/rjy5060/face.jpg'],x=0,y=100,w=100,h=100)

def bottle():
	#print("bottle")
	f = open('target.txt','w')
	f.write('s')
	f.close()   
	p.image_url(url=['http://personal.psu.edu/rjy5060/bottle.jpg'],x=0,y=100,w=100,h=100)
def start():
	print("Starting object detection...")
	p = Process(target=detect,args=(False,))
	p.start()
	#p.join()
def detect(serialwrite):
	if serialwrite:
		os.system("python test.py --serial yes")
	else:
		os.system("python test.py")

# add a button widget and configure with the call back
shutdownbutt = Button(label="SHUT DOWN")
shutdownbutt.on_click(shutdown)
facebutt = Button(label="Face Detect")
facebutt.on_click(face)
bottlebutt = Button(label="Bottles")
bottlebutt.on_click(bottle)
startbutt = Button(label="START")
startbutt.on_click(start)
# put the button and plot in a layout and add to the document
curdoc().add_root(column(row(shutdownbutt,facebutt,bottlebutt,startbutt),p))
