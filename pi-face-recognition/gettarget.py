import urllib.request
import time
import threading

exitFlag=0

def gettarget():
	while True:
		f = open('target.txt','r+')
		contents = urllib.request.urlopen("http://personal.psu.edu/kna5128/welcomepage.html").read()
		if str(contents) is not f.read():
			f.write(str(contents))
		f.close()
def readtarget():
	f = open('target.txt','r')
	target  = f.read()
	#print(target)
	#f.close()
	return target

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print ("Starting " + self.name)
      gettarget() #print_time(self.name, self.counter, 5)
      print ("Exiting " + self.name)
def print_time(threadName, delay, counter):
   while counter:
      if exitFlag:
         threadName.exit()
      time.sleep(delay)
      print ("%s: %s" % (threadName, time.ctime(time.time())))
      counter -= 1



