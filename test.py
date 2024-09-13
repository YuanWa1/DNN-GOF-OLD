from sim_utils5 import *
import matplotlib.pyplot as plt
import csv
import sys
import numpy as np
import tensorflow.keras as keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from dnn_inference.sig_test import split_test
import tensorflow as tf
#nN = 200, 500 , 1000 , 2000
nN = int(sys.argv[1])
max_hidden_unit = 18

x_train, y_train = generate_data(nN=nN)
y_train = y_train.T
x_train = x_train.T
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

# (x_train, y_train), (_, _) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz", 
#                                                                                   test_split=0.1)
# y_train, y_test = y_train[:,np.newaxis], y_test[:,np.newaxis]

# from sklearn import preprocessing
# scaler = preprocessing.MinMaxScaler()
# x_train = scaler.fit_transform(x_train)

n, d = x_train.shape
print('num of samples: %d, dim: %d' %(n, d))

from tensorflow import keras
from tensorflow.keras import layers

deg = 1/3
optimizer = 'sgd'
n,d_train = x_train.shape
drop_rate = 0.05
mult = 3/5

patience_nn = 20
min_delta = 0

def build_model_shallow():
  #n layers with same number of hidden units: max_hidden_unit
  n_h = np.floor(np.power(nN, deg)).astype(int)
  num_layers = np.floor(np.power(nN, deg)).astype(int)
  
  model = keras.Sequential()

  model.add(Dense(n_h, activation='relu', input_shape=(d_train,)))
  model.add(Dropout(drop_rate))

  for i in range(1, num_layers):
      model.add(Dense(n_h, activation='relu'))
      model.add(Dropout(drop_rate))

  model.add(Dense(1))


  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

def build_model_deep_nn1():
  #n layers with same number of hidden units: max_hidden_unit
  num_layers = np.floor(np.power(nN, deg)).astype(int)
  
  model = keras.Sequential()

  model.add(Dense(max_hidden_unit, activation='relu', input_shape=(d_train,)))
  model.add(Dropout(drop_rate))

  for i in range(1, num_layers):
      model.add(Dense(max_hidden_unit, activation='relu'))
      model.add(Dropout(drop_rate))

  model.add(Dense(1))


  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

def build_model_deep_nn2():
  #n layers with same number of hidden units: num_layers
  num_layers = np.floor(np.power(nN, mult*deg)).astype(int)
  
  model = keras.Sequential()

  model.add(Dense(num_layers, activation='relu', input_shape=(d_train,)))
  model.add(Dropout(drop_rate))

  for i in range(1, num_layers):
      model.add(Dense(num_layers, activation='relu'))
      model.add(Dropout(drop_rate))

  model.add(Dense(1))


  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model_null_shallow, model_alter_shallow = build_model_shallow(), build_model_shallow()

model_null_deep_nn1, model_alter_deep_nn1 = build_model_deep_nn1(), build_model_deep_nn1()

model_null_deep_nn2, model_alter_deep_nn2 = build_model_deep_nn2(), build_model_deep_nn2()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min',
					verbose=0, patience=20, restore_best_weights=True, min_delta=min_delta)

gamma = 0.9
batches = 20
tag = 0
n_train = np.floor(nN * gamma).astype(int)
n_test  = nN - n_train

batch_size = np.minimum(np.floor(n_train/batches).astype(int), np.floor(n_test/batches).astype(int)).astype(int)


fit_params = {'callbacks': [es],
			  #'epochs': 3000,
        'epochs': 200,
			  #'batch_size': 32,
        'batch_size': batch_size,
			  'validation_split': .2,
			  'verbose': 0}

## testing params
test_params = { 'split': "one-split",
                'inf_ratio': None,
                'perturb': None,
                'cv_num': 2,
                'cp': 'hommel',
                'verbose': 2}

## tuning params
tune_params = { 'num_perm': 100,
                'ratio_grid': [.2, .4, .6, .8],
                'if_reverse': 1,
                'perturb_range': 2.**np.arange(-3,3,.3),
                'tune_ratio_method': 'fuse',
                'tune_pb_method': 'fuse',
                'cv_num': 2,
                'cp': 'hommel',
                'verbose': 2}

# inf_feats = [np.arange(3), np.arange(5,11)]
inf_feats = [np.arange(1), np.arange(1,2)]

cue_shallow = split_test(inf_feats=inf_feats, model_null=model_null_shallow, model_alter=model_alter_shallow, eva_metric='mse')

cue_dnn1 = split_test(inf_feats=inf_feats, model_null=model_null_deep_nn1, model_alter=model_alter_deep_nn1, eva_metric='mse')

cue_dnn2 = split_test(inf_feats=inf_feats, model_null=model_null_deep_nn2, model_alter=model_alter_deep_nn2, eva_metric='mse')

P_value_shallow = cue_shallow.testing(x_train, y_train, fit_params, test_params, tune_params)

P_value_dnn1 = cue_dnn1.testing(x_train, y_train, fit_params, test_params, tune_params)

P_value_dnn2 = cue_dnn2.testing(x_train, y_train, fit_params, test_params, tune_params)

print("P_value_shallow: ", P_value_shallow)
print("P_value_dnn1: ", P_value_dnn1)
print("P_value_dnn2: ", )

import csv

def to_file(file_name, p_values_shallow, p_values_dnn1, p_values_dnn2):
    file_exists = False
    try:
        with open(file_name, 'r') as csvfile:
            file_exists = True
    except FileNotFoundError:
        pass
    
    with open(file_name, 'a', newline='') as csvfile:
        fieldnames = ['P_value_shallow', 'P_value_dnn1', 'P_value_dnn2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for i in range(0, len(p_values_shallow), 2):
            row = {}
            if i < len(p_values_shallow):
                row['P_value_shallow'] = f"{p_values_shallow[i]} {p_values_shallow[i+1]}" if i+1 < len(p_values_shallow) else p_values_shallow[i]
            if i < len(p_values_dnn1):
                row['P_value_dnn1'] = f"{p_values_dnn1[i]} {p_values_dnn1[i+1]}" if i+1 < len(p_values_dnn1) else p_values_dnn1[i]
            if i < len(p_values_dnn2):
                row['P_value_dnn2'] = f"{p_values_dnn2[i]} {p_values_dnn2[i+1]}" if i+1 < len(p_values_dnn2) else p_values_dnn2[i]
            writer.writerow(row)



file_name = f"output_for_{nN}.csv"

to_file(file_name,P_value_shallow, P_value_dnn1, P_value_dnn2)
