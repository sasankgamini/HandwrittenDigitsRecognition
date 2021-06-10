import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist

##print(trainimgs[0])
##plt.imshow(trainimgs[0])
##plt.show()
##cv2.imshow('testimg',trainimgs[0])
##cv2.waitKey()
##cv2.destroyAllWindows()

numimg1=cv2.imread('download.png',cv2.IMREAD_GRAYSCALE)
(thresh, numimg) = cv2.threshold(numimg1, 127, 255, cv2.THRESH_BINARY) #will only look at the number, not background by making background into 0's and the number as 1's(binary) 
numimg=cv2.bitwise_not(numimg)   #invert black and white background to match with data
numimg=cv2.resize(numimg,(28,28))
numimg=np.array([numimg])
numimg=numimg/255

((trainimgs, trainlabels),(testimgs,testlabels))= mnist.load_data()
class_names=[0,1,2,3,4,5,6,7,8,9]
trainimgs=trainimgs/255
testimgs=testimgs/255

print(numimg1)
##print(trainimgs[0])
##print(trainlabels[0])
##print(trainimgs.shape)
##print(testimgs.shape)

### One image
##plt.imshow(trainimgs[0])
##plt.xlabel(trainlabels[0])
##plt.show()

### Multiple images
##for n in range(0,8,1):
##    plt.subplot(4,2,n+1)
##    plt.imshow(trainimgs[n])
##    plt.xticks([])
##    plt.yticks([])
##    plt.ylabel(trainlabels[n])
##plt.show()
    
#building the model(blueprint)
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512,activation='relu'), #activation if it passes certian threshold
    keras.layers.Dense(10,activation='softmax') #gives percentages for each number in third layer
    ])

#Compile the model/properties of model(giving extra features/personalizing)
model.compile(optimizer='adam',    #different types of way for keras
              loss='sparse_categorical_crossentropy', #reduce loss to a low value
              metrics= ['accuracy'])   #to see if its working or not

#Train the model
model.fit(trainimgs,trainlabels,epochs=5)

#Test the model
test_loss, test_acc = model.evaluate(testimgs, testlabels)
print(test_acc)  #accuracy of test

#Predictions
predictions=model.predict(numimg)
print(predictions) #predictions of first test image
print(np.argmax(predictions)) #index of highest prediction

print(class_names[np.argmax(predictions)])

plt.imshow(numimg[0])
plt.show()

 
