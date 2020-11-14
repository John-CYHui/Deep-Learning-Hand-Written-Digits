import idx2numpy
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import datetime
from PIL import Image

data_file = 'C:/Users/hui94/Desktop/AI project/digit recognition/train-images.idx3-ubyte'
label_file = 'C:/Users/hui94/Desktop/AI project/digit recognition/train-labels.idx1-ubyte'

test_data_file = 'C:/Users/hui94/Desktop/AI project/digit recognition/t10k-images.idx3-ubyte'
test_label_file = 'C:/Users/hui94/Desktop/AI project/digit recognition/t10k-labels.idx1-ubyte'

data_arr = idx2numpy.convert_from_file(data_file)
label_arr = idx2numpy.convert_from_file(label_file)

test_arr = idx2numpy.convert_from_file(test_data_file)
test_label_arr = idx2numpy.convert_from_file(test_label_file)


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

#%%
#Train Neural Network Model
regular_lambda = 0.0001
model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape = (n,m)),
    tf.keras.layers.Dense(n*m, activation='relu',kernel_regularizer= keras.regularizers.L2(l= regular_lambda)),
    tf.keras.layers.Dense(400, activation='relu',kernel_regularizer= keras.regularizers.L2(l= regular_lambda)),
    tf.keras.layers.Dense(10)])

model.compile(optimizer = 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "C:/Users/hui94/Desktop/AI project/digit recognition/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(normal_arr,label_arr, epochs = 20,callbacks=[tensorboard_callback])

#%%

#Normalized Test Data
norm_test_arr = (test_arr - mean_)/range_
norm_test_arr = norm_test_arr / 255

#Reshape Test Data
test_len, n, m = norm_test_arr.shape

reshape_arr = np.zeros((test_len, n*m))

for i in range(test_len):
    reshape_arr[i] = np.reshape(norm_test_arr[i],(1,n*m))
    
norm_test_arr = reshape_arr

test_loss, test_acc = model.evaluate(norm_test_arr,  test_label_arr, verbose=2)

print('\nTest accuracy %:', test_acc * 100)


#%%
path = 'C:/Users/hui94/Desktop/AI project/digit recognition'
blank_arr = np.zeros((150,250))

final_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) #Convert the model to output probability for each digit
predictions = final_model.predict(norm_test_arr)

for i in range(20):
    idx = np.random.randint(0,10000-1)
    blank_arr[18:18+n,100:100+m] = test_arr[idx]
    im = Image.fromarray(blank_arr)
    im = im.convert("L")
    im.save(path + 'temp.jpeg')
    image = cv2.imread(path + 'temp.jpeg') 
    
    predict_val = str(np.argmax(predictions[idx]))
    actual_val = str(test_label_arr[idx])
    x,y,w,h = 0,50,100,75
    cv2.putText(image, 'Model Prediction: ' + predict_val, (x + int(w/10),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(image, 'Actual Number: ' + actual_val, (x + int(w/10),y + int(h/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
