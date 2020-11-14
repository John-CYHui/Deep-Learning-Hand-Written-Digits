# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:08:03 2020

@author: hui94
"""

import tensorflow as tf
import numpy as np
import idx2numpy
import cv2
from PIL import Image
from tensorflow import keras
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest
import time

data_file = 'C:/Users/hui94/Desktop/AI project/digit recognition/train-images.idx3-ubyte'
label_file = 'C:/Users/hui94/Desktop/AI project/digit recognition/train-labels.idx1-ubyte'

test_data_file = 'C:/Users/hui94/Desktop/AI project/digit recognition/t10k-images.idx3-ubyte'
test_label_file = 'C:/Users/hui94/Desktop/AI project/digit recognition/t10k-labels.idx1-ubyte'

data_arr = idx2numpy.convert_from_file(data_file)
label_arr = idx2numpy.convert_from_file(label_file)

test_arr = idx2numpy.convert_from_file(test_data_file)
test_label_arr = idx2numpy.convert_from_file(test_label_file)

# for i in range(5):
#     cv2.imshow('Image',test_arr[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

data_len, n, m = data_arr.shape


mean_ = data_arr.mean()
range_ = np.amax(data_arr) - np.amin(data_arr)

#Normalized Data
normal_arr = (data_arr - mean_)/range_
normal_arr = normal_arr / 255

#Reshape Data
reshape_arr = np.zeros((data_len, n*m))
for i in range(data_len):
    reshape_arr[i] = np.reshape(normal_arr[i],(1,n*m))
normal_arr = reshape_arr

#Normalized Test Data
norm_test_arr = (test_arr - mean_)/range_
norm_test_arr = norm_test_arr / 255

#Reshape Test Data
test_len, n, m = norm_test_arr.shape
reshape_arr = np.zeros((test_len, n*m))

for i in range(test_len):
    reshape_arr[i] = np.reshape(norm_test_arr[i],(1,n*m))
    
norm_test_arr = reshape_arr

X_train = normal_arr
y_train = label_arr
X_test = norm_test_arr
y_test = test_label_arr


# Parameters
num_steps = 100 # Total steps to train
num_classes = 10
num_features = n*m 
num_trees = 100 
max_nodes = 10000 

tf.reset_default_graph()

# Input and Target placeholders 
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int64, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes).fill()


# Build the Random Forest

forest_graph = tensor_forest.RandomForestGraphs(hparams)

# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))


# Start TensorFlow session

sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
time0 = time.time()
for i in range(1, num_steps + 1):

    _, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})

    if i % 1 == 0 or i == 1:

        acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})

        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
time1 = time.time()

train_time = time1-time0
print('Test Time:', train_time)

# Test Model
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))

#%%
path = 'C:/Users/hui94/Desktop/AI project/digit recognition/'
blank_arr = np.zeros((150,250))
for i in range(20):
    idx = np.random.randint(0,10000-1)
    blank_arr[18:18+n,100:100+m] = test_arr[idx]
    
    im = Image.fromarray(blank_arr)
    im = im.convert("L")
    im.save(path + 'temp.jpeg')
    image = cv2.imread(path + 'temp.jpeg') 
    
    X_temp = [X_test[idx]]
    #print('Model Prediction:', sess.run(tf.argmax(infer_op, 1), feed_dict={X: X_temp}), 'Actual:', [y_test[idx]])
    predict_val = str(sess.run(tf.argmax(infer_op, 1), feed_dict={X: X_temp})[0])
    actual_val = str(y_test[idx])
    x,y,w,h = 0,50,100,75
    cv2.putText(image, 'Model Prediction: ' + predict_val, (x + int(w/10),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(image, 'Actual Number: ' + actual_val, (x + int(w/10),y + int(h/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



