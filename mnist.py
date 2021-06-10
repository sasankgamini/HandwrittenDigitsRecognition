#importing modules/dependencies
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
import cv2

((trainimgs, trainlabels),(testimgs,testlabels))= mnist.load_data()
class_names=[0,1,2,3,4,5,6,7,8,9]
trainimgs=trainimgs/255
testimgs=testimgs/255


#opening image using opencv
cv2.imshow('testimg',trainimgs[0])
cv2.waitKey()
cv2.destroyAllWindows()

# One image using matplotlib
plt.imshow(trainimgs[0])
plt.xlabel(trainlabels[0])
plt.show()

# Multiple images
for n in range(0,25,1):
    plt.subplot(5,5,n+1)
    plt.imshow(trainimgs[n])
plt.show()
    
#building the model(blueprint)
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'), #activation if it passes certian threshold
    keras.layers.Dense(10,activation='softmax') #gives percentages for each number in third layer
    ])

#Compile the model/properties of model(giving extra features/personalizing)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics= ['accuracy'])

#Train the model
model.fit(trainimgs,trainlabels,epochs=5)

#saves a model so it will be faster and won't have to train each time you run it
model.save('savedDigitsTest.h5')

#Test the model
test_loss, test_acc = model.evaluate(testimgs, testlabels)
print(test_acc)  #accuracy of test

#Predictions
predictions=model.predict(testimgs)
print(predictions[1]) #predictions of first test image
print(np.argmax(predictions[1])) #index of highest prediction

print(class_names[np.argmax(predictions[1])])

plt.imshow(testimgs[1])
plt.show()


 
