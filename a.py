import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm
import numpy as np
import os 
from random import shuffle
import cv2

TEST_DIR = '/media/harsath/projects/medium/My projects/dogs vs cats/test'
TRAIN_DIR = '/media/harsath/projects/meduim/My projects/dogs vs cats/train'
LEARNING_RATE = 1e-3
MODEL_NAME = "dogsvscats-{}-{}.model".format(LEARNING_RATE,"6conv-fire")
IMAGE_SIZE = 50

def label_image(img):
    img_name = img.split(".")[-3]
if img_name == "cat":
return [1,0]
elif img_name == "dog":
return [0,1]

def train_data_loder():
    training_data = []
for img in tqdm(os.listdir(path=TRAIN_DIR)):
        img_lable = label_image(img)
        path_to_img = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_GRAYSCALE),(IMAGE_SIZE,IMAGE_SIZE))
        training_data.append([np.array(img),np.array(img_lable)])

    shuffle(training_data)
    np.save("training_data_new.npy",training_data)
return training_data

def testing_data():
    test_data = []
for img in tqdm(os.listdir(TEST_DIR)):
        img_labels = img.split(".")[0]
        path_to_img = os.path.join(TEST_DIR,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_GRAYSCALE),(IMAGE_SIZE,IMAGE_SIZE))
        test_data.append([np.array(img),np.array(img_labels)])

    shuffle(test_data)
    np.save("test_dataone.npy",test_data)
return test_data

import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')
#Conv Layer 1
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 2
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 3
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 4
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 5
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 6
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
#Fully Connected Layer with SoftMax as Activation Function
convnet = fully_connected(convnet, 2, activation='softmax')
#Regression for ConvNet with ADAM optimizer

convnet = regression(convnet, optimizer='adam', learning_rate=LEARNING_RATE, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

train = train_data_g[:-500]
test = train_data_g[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
Y = [i[1] for i in train]

#This is our Training data
test_x = np.array([i[0] for i in test]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
test_y = [i[1] for i in test]

model.fit(X, Y, n_epoch=6, validation_set=(test_x,  test_y), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

