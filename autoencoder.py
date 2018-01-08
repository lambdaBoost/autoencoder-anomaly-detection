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

#use the labelled data as a train/test set (note the different context of testing to normal for this model)
labelled_data=df[df['adshex'].isin(test_ident['adshex'])]
labelled_data=pd.merge(labelled_data,test_ident,on=['adshex','adshex'])
labelled_data['type']=labelled_data['type'].astype('category').cat.codes
labelled_data=labelled_data.drop(['adshex'],axis=1)

df=df[~df['adshex'].isin(test_ident['adshex'])]
df=df.drop(['adshex'],axis=1)

#preprocess - consider scaling/standardising at this point
train_set, test_set = train_test_split(labelled_data, test_size=0.2,random_state=57)
train_set = train_set[train_set['class'] == 'other']
train_set = train_set.drop(['class'], axis=1)
test_set = test_set.drop(['class'], axis=1)

train_set = train_set.values
test_set = test_set.values

#apply model to test set and evaluate by creating error sf from the predictions. 
#Take the top 60 or so errors and see how they match up with the removed adshexes (join with train data)

#define layers
input_dim = test_set.shape[1]
encoding_dim = int(input_dim/2)

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


nb_epoch = 25
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
history = autoencoder.fit(train_set,train_set,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test_set,test_set),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

autoencoder=load_model('model.h5')
                          
#predict on testing set
predictions=autoencoder.predict(test_set)







