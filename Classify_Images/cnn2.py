
# Description: This program classifies images

#Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

#Get the data types
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

#Get the shape of the data
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

#Take a look at the first image in the training data set
x_train[0]

#Show the image as a picture
import matplotlib.pyplot as plt
img = plt.imshow(x_train[0])

#Print the label of the first image of the training data set
print('The label is:', y_train[0])

#Print an example of the new labels, NOTE: The label 6 = [0,0,0,0,0,0,1,0,0,0]
print('The one hot label is:', y_train_one_hot[0])

#One Hot Encoding: Convert the labels into a set of 10 numbers to input into the neural network
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# print the new labels in the training data set
print(y_train_one_hot)

#Normalize the pixels in the images  to be values between 0 and 1
x_train = x_train / 255
x_test = x_test /255

#Build the CNN
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#Create the architecture
model = Sequential()

#Convolution layer to extract features from input image
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))

#Pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

#Convolution layer to extract features from input image
model.add(Conv2D(32, (5,5), activation='relu'))

#Pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

#Flatten the image layer
model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer = 'adam',
    metrics =['accuracy']
)

#Train the model
hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.3)

#Evaluate the model
model.evaluate(x_test, y_test_one_hot)[1]

#Visualize the models accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Visualize the models accuracy
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#Load data to make classifications
#from google.colab import files
#uploaded = files.upload()
my_image = plt.imread('cat.4014.jpg')

#Show the uploaded image
img = plt.imshow(my_image)

#Resize the image
from skimage.transform import resize
my_image_resized = resize(my_image, (32,32,3))
img = plt.imshow(my_image_resized)

#Get the probabilities for each class
import numpy as np
probabilities = model.predict( np.array([my_image_resized,]) )
#Show the probability for each class
probabilities

number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])

#Print the first 5 most likely classes / labels
print('Most likely class:', number_to_class[index[9]], "--Probability:", probabilities[0, index[9]])
print('Second most likely class:', number_to_class[index[8]], "--Probability:", probabilities[0, index[8]])
print('Third most likely class:', number_to_class[index[7]], "--Probability:", probabilities[0, index[7]])
print('Fourth most likely class:', number_to_class[index[6]], "--Probability:", probabilities[0, index[6]])
print('Fifth most likely class:', number_to_class[index[5]], "--Probability:", probabilities[0, index[5]])

#Save the model
model.save('my_model.h5')

#To load the model
from keras.models import load_model
model = load_model('my_model.h5')
