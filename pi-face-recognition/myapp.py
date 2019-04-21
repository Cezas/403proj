# myapp.py

#from random import random

from bokeh.layouts import column,row
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc, show
from cv2 import imread
#from scipy.misc import imread


#test = imread('bottle.jpg')
#test = test.flatten()
#print(test)
# create a plot and style its properties
p = figure(x_range=(0, 100), y_range=(0, 100))
p.image_url(url=['http://personal.psu.edu/rjy5060/othersentry.jpg'],x=0,y=100,w=100,h=100)
#p.image(image=[test],x=0,y=0,dw=10,dh=10)
#p.border_fill_color = 'black'
#p.background_fill_color = 'black'
#p.outline_line_color = None
#p.grid.grid_line_color = None


#p.image_url(url=["bottle.jpg"],x=500,y=500,w=500,h=500)
# add a text renderer to our plot (no data yet)
#r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
#           text_baseline="middle", text_align="center")

#i = 0

#ds = r.data_source

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

 #   global i

    # BEST PRACTICE --- update .data in one step with a new dict
 #   new_data = dict()
 #   new_data['x'] = ds.data['x'] + [random()*70 + 15]
 #   new_data['y'] = ds.data['y'] + [random()*70 + 15]
 #   new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
 #   new_data['text'] = ds.data['text'] + [str(i)]
 #   ds.data = new_data

  #  i = i + 1

# add a button widget and configure with the call back
shutdownbutt = Button(label="SHUT DOWN")
shutdownbutt.on_click(shutdown)
facebutt = Button(label="Face Detect")
facebutt.on_click(face)
bottlebutt = Button(label="Bottles")
bottlebutt.on_click(bottle)

# put the button and plot in a layout and add to the document
curdoc().add_root(column(row(shutdownbutt,facebutt,bottlebutt),p))
