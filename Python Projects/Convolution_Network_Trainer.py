'''
SBrnBio
Convolution Network

Modified off of the following code: https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
'''
#--------------------------------------------
#### Library Packages
#--------------------------------------------
from __future__ import division, print_function, absolute_import

import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from scipy.misc import imresize

from sklearn.model_selection import train_test_split

from glob import glob

import tflearn

from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import to_categorical

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

from tflearn.metrics import Accuracy
#--------------------------------------------
#### Import picture files
#--------------------------------------------

#file path to the train folder
#update with the proper directory
files_path = 'C:/.../Convolution_Net/data/train/'

#be sure to fill the cat and dog folders with kaggle data
cat_files_path = os.path.join(files_path, 'cat', '*.jpg')
dog_files_path = os.path.join(files_path, 'dog', '*.jpg')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))

n_files = len(cat_files) + len(dog_files)

#Check to see number of files
print('Number of Files: %d' % n_files)

#standardize the size of all the images
image_size = 32

allX = np.zeros((n_files, image_size, image_size, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0

i = 0
for f in cat_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (image_size, image_size, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue


for f in dog_files:
    try:

        img = io.imread(f)
        new_img = imresize(img, (image_size, image_size, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue

#--------------------------------------------
#### Preparation of the samples
#--------------------------------------------
        
# test-train split   
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)

print("Transforming Images")

#--------------------------------------------
#### Transforming the Images
#--------------------------------------------

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

#--------------------------------------------
#### Defining the network architecture
#--------------------------------------------
print('Defining the Network')

# Input is a 32x32 image with 32 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 16, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 32 filters
conv_2 = conv_2d(network, 32, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 32 filters
conv_3 = conv_2d(conv_2, 32, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 256 node layer
network = fully_connected(network, 256, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name='Accuracy')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model/catdog_model.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='model/tmp/tflearn_logs/')

print('Network Definition Complete')

#--------------------------------------------
#### Train model for N epochs
#--------------------------------------------
print('Preparing to Train Model')

model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=30,
      n_epoch=20, run_id='catdog_model', show_metric=True)


print('Saving Model')
model.save('model/catdog_model_final.tflearn')

#--------------------------------------------
#### Loading the model
#--------------------------------------------
print('Loading Model')

model.load('model/catdog_model_final.tflearn',)

#--------------------------------------------
#### Test Image
#--------------------------------------------
'''
test_dog_1.jpg should be a .jpg image that you desire
for the model to predict
'''
from PIL import Image
img = Image.open('test_dog_1.jpg')
plt.imshow(img)

data = img.resize((image_size, image_size), Image.NEAREST)
plt.imshow(data)

#--------------------------------------------
#### Prediction
#--------------------------------------------
prediction = model.predict([data])[0]

print(f"Dog: {prediction[0]}, Cat: {prediction[1]}")
