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
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

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
df_adshex=df['adshex']
df=df.drop(['adshex'],axis=1)

#make a training and testing set and process
#use the labelled data to test, use some of the unlabelled data to train (assume that very few entries will be of class surveil in the unlabelled data)
#use 10% of the input data as a training set
df,train_set=train_test_split(df,test_size=0.1,random_state=57)

#also consider adding some of the actual data to the train/test set to improve the size
test_set = labelled_data

#save test set labels for later
test_set_labels=test_set['class']
#get number of positive classes in test set
test_set_positives=len(test_set[test_set['class']=='surveil'])
test_set = test_set.drop(['class'], axis=1)

#convert to array and normalise
train_set = preprocessing.MinMaxScaler().fit_transform(train_set.values)
test_set = preprocessing.MinMaxScaler().fit_transform(test_set.values)


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


nb_epoch = 100
batch_size = 50
autoencoder.compile(optimizer='Adamax', 
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
rmse = pow(np.mean(np.power(test_set - predictions, 2), axis=1),0.5)
error_table = pd.DataFrame({'reconstruction_error': rmse,
                        'actual_class': test_set_labels})
#we know how many entries have a positive in the test set. Take this number and label the predictions with the highest rmse (ie the outliers) with the positiveclass prediction
error_table['predicted_class'] = np.where(error_table['reconstruction_error'] >= min(error_table.nlargest(int(test_set_positives),'reconstruction_error', keep='first')['reconstruction_error']), 'surveil', 'other')

#get confusion matrix
print(confusion_matrix(error_table['actual_class'],error_table['predicted_class']))



#plot error
groups=error_table.groupby('actual_class')
fig,ax=plt.subplots()
for name, group in groups:
    ax.plot(group.index,group.reconstruction_error,marker='o', ms=3.5, linestyle='',
            label= name)