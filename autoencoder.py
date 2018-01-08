# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:20:16 2018

@author: alex.hall
"""

#script to find anomalies through an autoencoder NN in tensorflow
#uses flight dataset by default to identify potential spyplanes - https://www.kaggle.com/jboysen/spy-plane-finder

#import libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler

#import dataset
df=pd.read_csv("C:\\Users\\Alex\\Documents\\datasets\\spy-plane-finder\\planes_features.csv")

#convert type column to integer (there are probably better approaches to use here)
df['type']=df['type'].astype('category').cat.codes

#import labelled aircraft
test_ident = pd.read_csv("C:\\Users\\Alex\\Documents\\datasets\\spy-plane-finder\\train.csv")

#get a 'testing set' and leave the rest of the data for actual analysis (note the different context of testing to normal for this model)
test_set=df[df['adshex'].isin(test_ident['adshex'])]
df=df[~df['adshex'].isin(test_ident['adshex'])]

#preprocess - consider scaling/standardising at this point
unlabelled_test=test_set.drop(['adshex']) #drop labels
df=df.drop(['adshex'])

#apply model to test set and evaluate by creating error sf from the predictions. 
#Take the top 60 or so errors and see how they match up with the removed adshexes (join with train data)

#define layers
input_dim = unlabelled_test.shape[1]
encoding_dim = int(input_dim/2)

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


nb_epoch = 100
batch_size = 32
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(unlabelled_test, unlabelled_test,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
#apply model to df and see what we get







