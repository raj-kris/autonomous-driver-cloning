import os
import csv

samples = []
# folder_path = '../t_data/c1/'
## My training data is stored in different folders for different drivers 
## It has 2 -3 laps of normal driving
## It has 2-3 laps of reverse driving around the track
## Few laps of side to center driving
## few laps of driving on the alternate track to make the driving more general
folder_path_list = os.listdir('../t_data/')
folder_base_path = '../t_data/'

for folder_path in folder_path_list: 
    folder_path = folder_base_path + folder_path
    print(folder_path)
    with open(folder_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:            
            samples.append(line)
            

print(len(samples))
## Train and test split of the drive log lines read from all the csv files.
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import random

## Use of a generator with yield key word to save on memory required.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # name = folder_path + 'IMG/'+batch_sample[0].split('/')[-1]
                name = batch_sample[0]                
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Code for Augmentation of data.
                # We take the image and just flip it and negate the measurement
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function to minimize memory use
# The generator rather than loading all the images at once, proceedes with loading batches
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

# Build the Keras Model

from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten , Dropout

# The network I choose is the NVIDIA network explained in the tutorial video

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(row,col,ch),output_shape=(row,col,ch)))

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))    
          
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
          
model.add(Convolution2D(64,3,3,activation="relu"))
          
model.add(Convolution2D(64,3,3,activation="relu"))
          
model.add(Flatten())
          
model.add(Dense(100))
model.add(Activation("relu"))

#Adding a dropout layer to avoid overfitting.
model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Activation("relu"))
          
model.add(Dense(10))
model.add(Activation("relu"))

model.add(Dense(1))

          
import math

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

model.save('model.h5')

print("Model saved")

model.summary()
          
          
