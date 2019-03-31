from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import graph_util #custom code from online
#import matplotlib.pyplot as plt

#print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#PICK AND CHOOSE
#====================================
tempimages = []
templabels = []
temptestimages=[]
temptestlabels=[]

for i in range(len(train_labels)):
	if train_labels[i]==0 or train_labels[i]==7:
		tempimages.append(train_images[i])
		templabels.append(train_labels[i])
train_images = np.asarray(tempimages)
train_labels = np.asarray(templabels)

for i in range(len(test_labels)):
        if test_labels[i]==0 or test_labels[i]==7:
                temptestimages.append(test_images[i])
                temptestlabels.append(test_labels[i])
test_images = np.asarray(temptestimages)
test_labels = np.asarray(temptestlabels)
train_images = train_images / 255.0
test_images = test_images / 255.0
#================================================
'''
#-------------------------------------Define the model in Keras
model = keras.models.Sequential()


model.add(keras.layers.Dense(32,input_shape=(28,28)))
model.add(keras.layers.Activation('relu'))

a,b,c  = model.output_shape
a = b*c
print(a)

model.add(keras.layers.Permute([1, 2]))  # Indicate NHWC data layout
model.add(keras.layers.Reshape((a,)))

model.add(keras.layers.Dense(32))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))
#-------------------------------------------------------------------
'''

#model = keras.Sequential()
#model.add(keras.layers.Dense(1,input_shape=(28,28), activation='softmax'))
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)
model.save('testmod.h5')
new_model = keras.models.load_model('testmod.h5')
loss, acc = new_model.evaluate(test_images, test_labels)



#----------------------Serialize the graph
#sess = keras.backend.get_session()
#constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['activation_3/Softmax'])
#tf.train.write_graph(constant_graph, "", "graph.pb", as_text=False)
#----------------------------------------







'''
#CURRENTLY UNUUSED
#---------------attempted good pb------------------
sess = keras.backend.get_session()
graph_def = sess.graph.as_graph_def(add_shapes=True)
graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [model.output.name.split(':')[0]])
graph_util.make_cv2_compatible(graph_def)
tf.train.write_graph(graph_def, ".", 'tf_model.pb', as_text=False)
tf.train.write_graph(graph_def, ".", 'tf_model.pbtxt', as_text=True)
#--------------------------------------------------
'''
'''
#THIS HAS BEEN MOVED TO THE KERAS CONVERTER SCRIPT
#---------------------------------------------------------------
with tf.gfile.FastGFile('tf_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

# Strip Const nodes.
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'Const':
        del graph_def.node[i]
    for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                  'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                  'Tpaddings']:
        if attr in graph_def.node[i].attr:
             del graph_def.node[i].attr[attr]

# Save stripped model.
tf.train.write_graph(graph_def, ".", 'tf_model.pbtxt', as_text=True)
#-----------------------------------------------------
'''
