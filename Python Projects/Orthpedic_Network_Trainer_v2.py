'''
SethBrnBio
Deep Neural Network
'''
#--------------------------------------------
#### Library Packages
#--------------------------------------------

from __future__ import print_function
import numpy as np
import tflearn

#--------------------------------------------
#### Load csv file
#--------------------------------------------

# Load the csv file, the csv in question has been modified so that 
# 'Hernia' = 2, 'Spondylolisthesis' = 1 and 'Normal' = 0 
from tflearn.data_utils import load_csv
data, labels = load_csv('column_3C_weka - Copy.csv',
                        categorical_labels=True, n_classes=3)


data =  np.array(data, dtype=np.float32)

#--------------------------------------------
#### Building the model
#--------------------------------------------
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net)

#--------------------------------------------
#### Defining the model
#--------------------------------------------
model = tflearn.DNN(net, checkpoint_path='model/orthopedic_model_3C.tflearn', 
                    max_checkpoints = 3, tensorboard_verbose = 3,
                    tensorboard_dir='model/tmp/tflearn_logs/')

#--------------------------------------------
#### Train model for N epochs
#--------------------------------------------
model.load('model/orthopedic_model_3C.tflearn')

model.fit(data, labels, n_epoch=100,  run_id='oprthopedic_model_3C', 
          batch_size=10, show_metric=True)

#--------------------------------------------
#### Saving and loading the model
#--------------------------------------------
model.save('model/orthopedic_model_3C.tflearn')
model.load('model/orthopedic_model_3C.tflearn')

#--------------------------------------------
#### Predicting with the model
#--------------------------------------------

#Hernia
John = [37.686,4.010,42.948,30.675,85.241,1.664]

#Spondylolisthesis
Jake = [66.536,23.157,46.775,40.378,137.440,15.378]

pred = model.predict([John, Jake])
print("\n")
print("John")
print("Normal:", pred[0][0])
print("Spondylolisthesis:", pred[0][1])
print("Hernia:", pred[0][2])
print("\n")
print("Jake")
print("Normal:", pred[1][0])
print("Spondylolisthesis:", pred[1][1])
print("Hernia:", pred[1][2])