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

#import dataset
df=pd.read_csv("C:\\Users\\alex.hall\\Documents\\datasets\\spy-plane-finder\\planes_features.csv")

#import labelled aircraft
test_ident = pd.read_csv("C:\\Users\\alex.hall\\Documents\\datasets\\spy-plane-finder\\train.csv")

#get a 'testing set' and leave the rest of the data for actual analysis (note the different context of testing to normal for this model)
test_set=df[df['adshex'].isin(test_ident['adshex'])]
df=df[~df['adshex'].isin(test_ident['adshex'])]