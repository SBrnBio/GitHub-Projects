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
# 'Abnormal' = 1 and 'Normal' = 0 
from tflearn.data_utils import load_csv
data, labels = load_csv('column_2C_weka - Copy.csv',
                        categorical_labels=True, n_classes=2)


data =  np.array(data, dtype=np.float32)

#--------------------------------------------
#### Building the model
#--------------------------------------------
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

#--------------------------------------------
#### Defining the model
#--------------------------------------------
model = tflearn.DNN(net, checkpoint_path='model/orthopedic_model_2C.tflearn', 
                    max_checkpoints = 3, tensorboard_verbose = 3,
                    tensorboard_dir='model/tmp/tflearn_logs/')

#--------------------------------------------
#### Train model for N epochs
#--------------------------------------------
#model.load('model/orthopedic_model_2C.tflearn')

model.fit(data, labels, n_epoch=100,  run_id='oprthopedic_model_2C', 
          batch_size=10, show_metric=True)

#--------------------------------------------
#### Saving and loading the model
#--------------------------------------------
model.save('model/orthopedic_model_2C.tflearn')
model.load('model/orthopedic_model_2C.tflearn')

#--------------------------------------------
#### Predicting with the model
#--------------------------------------------

# Abnormal
John = [62.01,21.02,38.06,42.47,100.67,0.27]
#normal
Jake = [60.16,12.56,45.20,42.60,119.03,3.83]

pred = model.predict([John, Jake])
print("\n")
print("John Normality Probability:", pred[0][0])
print("John Abnormality Probability:", pred[0][1])
print("\n")
print("Jake Normality Probability:", pred[1][0])
print("Jake Abnormality Probability:", pred[1][1])