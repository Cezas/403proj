#huge creds to pyimagesearch for the boilerplate code of basic detection and tracking
# USAGE
#python test.py --cascade cascades/haarcascade_frontalface_default.xml --encodings encodings.pickle --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --serial no

#TODOTODOTODOTODOTODOTODOTODOTODO

###BIG IDEA

#idea: adjustlastgoodbox to have lockon manner as resize
#PRIORITY: optimize detection time (esp for ssd)


#refactor and clean up
	#move names from object detect to tracker
#possibly incorporate a hard refresh of the tracker as it can get stuck on differnt things (on Mosse)
	#accomplish this by checking "unknown" vs "dataset"
#incorporate multiple haar cascade xmls
#object must not move too much during detection process,or else tracker wont lock on
#save previous location/appearance of tracked object and restablish that as new target 

#TODOTODOTODOTODOTODOTODOTODOTODO

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import serial
import cv2
#from multiprocessing import Process
#from multiprocessing import Queue
import numpy as np


#import gettarget as targ
###############functions##########################
def inittracker():
	tracker = cv2.TrackerMOSSE_create()
	initBB = None #initial bounding box that will encapsulate the object
	lastgoodbox = None
	initialized = False #denotes on whether or the tracker was initialized

	return (tracker,initBB,lastgoodbox,initialized)



def ssddetect(frame,net,CLASSES,IGNORE): #only returns 1st detection for now
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (200, 200)),                0.007843, (200, 200), 127.5)
	#cv2.resize(frame, (300, 300))
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	#inp = np.random.standard_normal([1, 28, 28, 1]).astype(np.float32)
	#net.setInput(inp.transpose(0, 3, 1, 2))
	detections = net.forward()
	
	rects = []	

	#print(detections.shape[2])	
	# loop over the detections
	if detections.shape[2] == 1:
		return rects
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
		if confidence > .5:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			if CLASSES[idx] in IGNORE:
				continue

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			rects.append(box)			
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
        			confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
        			COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
        			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			return rects	
	#print(rects)
	return rects			



##############END FUNCTIONS	




if __name__=="__main__":		
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-s", "--serialwrite", required=False,
                help="say yes if writing to serial")
	
	args = vars(ap.parse_args())
	
	#**********INITIALIZING TRACKER & VARIOUS VARS
	serialwrite = False
	if args['serialwrite']=='yes':
		serialwrite = True
	ser = serial.Serial(    
	port='/dev/ttyS0',
	baudrate = 9600,
	parity=serial.PARITY_NONE,
	stopbits=serial.STOPBITS_ONE,
	bytesize=serial.EIGHTBITS,
	timeout=1
	)

	target = ' '
	widthsize = 350
	(tracker,initBB,lastgoodbox,initialized) = inittracker()
	success = False #denotes on whether or not a detection occurred
	detectrate = 100000
	redetectrate = 100000
	#detectrate = 270  #how many frames pass until object detection force checks again
	starttime = None #initially none, will be manually determined thru user input
	usingSSD = False 
	stuckcounter = 0
	#*********END TRACKER INIT
	
	# load the known faces and embeddings along with OpenCV's Haar
	# cascade for face detection
	print("[INFO] loading encodings + face detector...")
	data = pickle.loads(open('encodings.pickle', "rb").read())
	detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
	
	#*********LOADING NET

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	
	#CLASSES = ["T-shirt/top","Sneaker"]
	#IGNORE = ["background"]
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
	IGNORE =  ["background", "aeroplane", "bicycle", "bird", "boat",
         "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "pottedplant","person", "sheep",
        "sofa", "train", "tvmonitor"]

	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	print("[INFO] loading model...")
	
	#net = cv2.dnn.readNetFromTensorflow('sorted_inference_graph.pb','frost.pbtxt')

	net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel') #default
	#*********END NET

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()	
	time.sleep(2.0)	
	framecounter = 0
	
	#start pinging the website for the target
	#thread1 = targ.myThread(1, "Thread-1", 1)
	# Start new Threads
	#thread1.start()
	#thread1.join()


#********************MAIN ALGO********************
	# loop over frames from the video file stream
	while True:
		#constantly read the target file (incase it changes)
		#target = targ.readtarget()
		#print(target)
		framecounter = framecounter + 1
		frame = vs.read()
		frame = imutils.resize(frame, width=widthsize)
		(H, W) = frame.shape[:2]  #acquire the height and width of the frame and put it into a tuple for later use 	
		innerframe = None
		innerrects = None
		# check to see if we are currently tracking an object
		if initBB is not None:
			# grab the new bounding box coordinates of the object
			#print("right before update", success)
			(success, box) =  tracker.update(frame) #does this happen because its too laggy?  frame differences are huge? 
			#print("TRACKER gives", box)
                	# check to see if the tracking was a success
			#forced redetect on the current tracking frame

			#print(success,lastgoodbox,framecounter,redetectrate)
			if success and framecounter%redetectrate==0 and framecounter >= redetectrate:
				#print("redetecting")
				(x, y, w, h) = [int(v) for v in initBB] 
				innerframe = frame[y:y+h, x:x+w]
				gray = cv2.cvtColor(innerframe, cv2.COLOR_BGR2GRAY)
				# detect faces in the grayscale frame
				innerrects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                        minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
				
				if len(innerrects) != 0:
					(a, b, c, d) = [int(v) for v in innerrects[0]]
					(tracker,initBB,lastgoodbox,initialized) = inittracker()
					initBB = (x+a,y+b,c,d)
					initialized = tracker.init(frame,initBB)
				else:
					success = False
					lastgoodbox = None
			if success:
				(x, y, w, h) = [int(v) for v in box] #extracting boxes' dimensions and position
				if lastgoodbox == (x,y,w,h):
					stuckcounter = stuckcounter + 1
					if stuckcounter >= 100:
						success = False
					print("stuckcounter =",stuckcounter)
				else:
					stuckcounter=0	
				lastgoodbox = (x,y,w,h)
				centroidx = x+w/2
				centroidy= y+h/2
				#print(centroidx, widthsize*.9, widthsize*.1)
				cv2.rectangle(frame, (x, y), (x + w, y + h),
                              	(0, 255, 0), 2) #frame,top left, bottom right, color, thicness
				if centroidx>=.9*widthsize or centroidx<=.1*widthsize or centroidy>=.9*widthsize or centroidy<=.1*widthsize  :
					(tracker,initBB,lastgoodbox,initialized) = inittracker()
					success = False
					
				#************serial writing, make sure to convert to 0-255 range
				if serialwrite:
					scalingfactor = 255/widthsize
					print(centroidx,centroidy,w,h)
					#print('converted centx',int(centroidx*scalingfactor))
					#print('converted centy',int(centroidy*scalingfactor))
					#print('converted w',int(w*scalingfactor))
					#print('converted h',int(h*scalingfactor))
	
					ser.write(b'O')
					ser.write(chr(int(centroidx*scalingfactor)).encode())
					ser.write(chr(int(centroidy*scalingfactor)).encode())
					ser.write(chr(int(w*scalingfactor)).encode())
					ser.write(chr(int(h*scalingfactor)).encode())
					ser.write(b'C')
					time.sleep(.1)
			
			#disable relockon for now		
			#elif lastgoodbox is not None:
			#	print("LOCKON")
			#	tempbox = lastgoodbox
			#	(tracker,initBB,lastgoodbox,initialized) = inittracker()
			#	initBB = tempbox
			#	initialized = tracker.init(frame,initBB)
			#	print("USING COORDS",initBB)
			#	cv2.rectangle(frame, (x, y), (x + w, y + h),
                        #       (0, 255, 0), 2) #frame,top left, bottom right, color, thicness
			#	success = True							
			else:
				(tracker,initBB,lastgoodbox,initialized) = inittracker()
				lastgoodbox = None
			#**************

		#*******attempt object recognition every N frames and when target is undetected      
		if framecounter%detectrate==1 or not success: #I use %==1 as %==0 would have this go off a lot initially
			#print('#######ATTEMPTING DETECTION##########')	
			stuckcounter = 0	
			if usingSSD:
				rects = ssddetect(frame,net,CLASSES,IGNORE)
				#print(len(rects))
				if len(rects):
					#************once object has been detected, lockon with tracker
					(x, y, x2, y2) = [int(v) for v in rects[0]] #only takes first thing detected for now
					initBB = (x,y,x2-x,y2-y) #this value of initBB will be used for the tracker input
					#print('***********RECOGNIZE ',name,' *****************' )
					#print("face detect gives ",initBB)             
					if not initialized:     
						initialized = tracker.init(frame,initBB)
						(success, box) = tracker.update(frame)
						print("should be end of detect looop", success)
						#print(initialized)
					#******************************
			else:

				# convert the input frame from (1) BGR to grayscale (for face
				# detection) and (2) from BGR to RGB (for face recognition)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					
				# detect faces in the grayscale frame
				rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
					minNeighbors=5, minSize=(30, 30),
					flags=cv2.CASCADE_SCALE_IMAGE)
		
				# OpenCV returns bounding box coordinates in (x, y, w, h) order
				# but we need them in (top, right, bottom, left) order, so we
				# need to do a bit of reordering
				boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
				
				# compute the facial embeddings for each face bounding box
				encodings = face_recognition.face_encodings(rgb, boxes)
				names = []
 		
				# loop over the facial embeddings	
				for encoding in encodings:
					# attempt to match each face in the input image to our known
					# encodings
					matches = face_recognition.compare_faces(data["encodings"],encoding)
					name = "Unknown"
		
					#********match detected, mark time
					if starttime is not None:
						print('Face detected in',round(time.time()-starttime,2),'seconds')
						starttime = None
					#**********
		
					
					# check to see if we have found a match
					if True in matches:
						
						#iterare through the				
						# dictionary to count the total number of times each face
						# was matched
						matchedIdxs = [i for (i, b) in enumerate(matches) if b]
						counts = {}
		
						# loop over the matched indexes and maintain a count for
						# each recognized face face
						for i in matchedIdxs:
							name = data["names"][i]
							counts[name] = counts.get(name, 0) + 1
		
						# determine the recognized face with the largest number
						# of votes (note: in the event of an unlikely tie Python
						# will select first entry in the dictionary)
						name = max(counts, key=counts.get)
		
					# update the list of names
					names.append(name)
					# loop over the recognized faces
					for ((top, right, bottom, left), name) in zip(boxes, names):
						# draw the predicted face name on the image
						cv2.rectangle(frame, (left, top), (right, bottom),
							(0, 255, 0), 2)
						y = top - 15 if top - 15 > 15 else top + 15
						cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
							0.75, (0, 255, 0), 2)
			
					#************once object has been detected, lockon with tracker
					#---will need to decide targets before this point, 
					#initBB = boxes[0]
					(x, y, w, h) = [int(v) for v in rects[0]] #only takes first thing detected for now
					initBB = (x,y,w,h) #this value of initBB will be used for the tracker input
					print('***********RECOGNIZE ',name,' *****************' )
					#print("face detect gives ",initBB)		
					if not initialized:	
						initialized = tracker.init(frame,initBB)
						(success, box) = tracker.update(frame)
						#(success, box) = tracker.update(frame)
						#print(initialized)
					#******************************
			
		# display the image to our screen
		cv2.imshow("Frame", frame)
		#key = input()
		key = cv2.waitKey(1) & 0xFF	
		
		f = open('target.txt','r')
		tempkey = f.read()
		tempkey = ord(tempkey[0])
		#print(tempkey,key)
		if tempkey != 10:
			#print(tempkey)
			key=tempkey		

		if key == ord("s") and not usingSSD:
			print('Switching targets, using SSD')
			usingSSD = True
			(tracker,initBB,lastgoodbox,initialized) = inittracker()
			success = False
		if key == ord("h") and usingSSD:
			print('Switching targets, face detect')
			usingSSD = False
			(tracker,initBB,lastgoodbox,initialized) = inittracker()
			success = False
		#*****timer start******
		if key == ord("t"):
			print('Starting timer for detection')
			starttime=time.time()
		#*******************
	

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			f = open('target.txt','w')
			f.write('')
			f.close()
			break
	
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	
