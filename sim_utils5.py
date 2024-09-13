import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from math import pi
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.regularizers import L1
from keras import layers
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import cauchy

#-------------------------------------------------------------------------------------------------------------------#
def generate_data(nN = 100, n_x_true = 1, n_y_true = 1, low = -1, high = 1):
    X_mat = np.random.uniform(low = low, high = high, size = nN*(n_x_true+1)).reshape((n_x_true+1, nN))
    Y_true = np.cos(2 * pi * X_mat[1,:])
    return X_mat, Y_true.reshape((1, nN))
    
#-------------------------------------------------------------------------------------------------------------------#
def get_linear_reg_p_val(X, Y, beta0 = 0, tail = "two"):
    n,p = X.shape
    df = n-p-1
    X = np.hstack((np.ones((n,1)), X))
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, Y)
    XtX_inv = np.linalg.inv(XtX)
    C = np.diagonal(XtX_inv).reshape((p+1,1))
    beta_hat = np.dot(XtX_inv, Xty)
    Y_hat = np.dot(X, beta_hat)
    MSE = np.sum(np.power(Y-Y_hat, 2))/df
    t_score = (beta_hat - beta0)/np.sqrt(MSE * C)
    if tail == "two":
        p_value = t.sf(np.abs(t_score), df) * 2  # for two-tailed test
    elif tail == "left":
        p_value = t.sf(t_score, df)  # for left-tailed test
    elif tail == "right":
        p_value = t.sf(-t_score, df)  # for right-tailed test
    else:
        raise ValueError(
            "Invalid tail argument. Use 'two', 'left', or 'right'.")
    return p_value.reshape(p+1,)

#-------------------------------------------------------------------------------------------------------------------#
def scheduler(epoch, lr):
     if epoch < 50:
        return 0.1
     else:
        return 0.1 / np.log(np.exp(1)+epoch)
#-------------------------------------------------------------------------------------------------------------------#
def calculate_kappa(Y_train, predict_train,
                    plus = False):
    if plus:
        kappa = np.mean(np.power(Y_train - predict_train.T, 4))
    else:
        mse_train = np.mean(np.power(Y_train - predict_train.T, 2))
        kappa = np.mean(np.power(Y_train - predict_train.T, 4)) - np.power(mse_train, 2)
        
    return kappa
#-------------------------------------------------------------------------------------------------------------------#
#######################################
# Fit a shallow relu network
# training data
# X is a dxn matrix with n being the training sample size
# Y is a 1xn vector of responses
#
# n_h is the number of hidden units
#######################################
def fit_shallow_relu_nn(X, Y, n_h, 
                       optimizer = 'adam', epochs = 1000, batch_size = 1,
                       early_stopping = True,
                       validation_split = 0.2, patience = 5, min_delta = 1e-4,
                       verbose = 0, drop_rate = 0.2):
    d_train,n = X.shape

    callback_train = EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta,
                                   restore_best_weights=True)
    callback_lr = LearningRateScheduler(scheduler)
    model_ReLU_nn = keras.Sequential()
    model_ReLU_nn.add(layers.InputLayer(input_shape=(d_train,)))
    model_ReLU_nn.add(layers.Dense(n_h, activation = "relu"))
    model_ReLU_nn.add(layers.Dropout(drop_rate))
    model_ReLU_nn.add(layers.Dense(1))

    model_ReLU_nn.compile(optimizer=optimizer, loss='mse')

    if early_stopping:
        model_ReLU_nn.fit(X.T, Y.T, batch_size=batch_size, epochs=epochs, 
                          validation_split = validation_split, callbacks = [callback_train, callback_lr],
                          verbose = verbose)
    else:
        model_ReLU_nn.fit(X.T, Y.T, batch_size=batch_size, epochs=epochs, verbose = verbose,
                          callbacks = [callback_lr])
    
    predict_train = model_ReLU_nn.predict(X.T, verbose = verbose)
    mse_train = np.mean(np.power(Y - predict_train.T, 2))

    return mse_train, predict_train
    
#-------------------------------------------------------------------------------------------------------------------#
#######################################
# Fit a deep relu network
# training data
# X_train is a dxn matrix with n being the training sample size
# Y_train is a 1xn vector of responses
#
# testing data
# X_test is a dxm matrix with m being the testing sample size
# Y_test is a 1xm vector of responses
#
# n_h is a vector containing the number of hidden units in each layer
# num_layers is the number of layers in the deep network
#######################################
def fit_deep_relu_nn(X_train, Y_train, n_h, num_layers, 
                     optimizer = 'adam', epochs = 1000, batch_size = 1,
                     early_stopping = True,
                     patience = 5, validation_split = 0.2, min_delta = 1e-4,
                     verbose = 0, drop_rate = 0.2):
    d_train,n = X_train.shape
    callback_train = EarlyStopping(monitor='val_loss', 
                                   patience=patience,
                                   min_delta=min_delta,
                                   restore_best_weights = True)
    callback_lr = LearningRateScheduler(scheduler)
        
    input = Input(shape=(d_train,))
    inp = input
    for i in range(num_layers):
        x = Dense(n_h[i], activation = 'relu')(inp)
        x = layers.Dropout(drop_rate)(x)
        inp = x
    output = Dense(1)(x)
    model_ReLU_dnn = Model(input, output)

    model_ReLU_dnn.compile(optimizer = optimizer, loss = 'mse')

    if early_stopping:
        model_ReLU_dnn.fit(X_train.T, Y_train.T, batch_size = batch_size, epochs = epochs,
                          validation_split = validation_split, callbacks = [callback_train, callback_lr],
                          verbose = verbose)
    else:
        model_ReLU_dnn.fit(X_train.T, Y_train.T, batch_size = batch_size, epochs = epochs,
                           verbose = verbose, callbacks = [callback_lr])
        
    predict_train = model_ReLU_dnn.predict(X_train.T, verbose = verbose)
    mse_train = np.mean(np.power(Y_train - predict_train.T, 2))
    
    return mse_train, predict_train
    
#-------------------------------------------------------------------------------------------------------------------#
#############################
# This function is used to find the p-value for the DNN-GOF test
# mse_train, mse_test are the MSEs for the training data and testing data
# n_train, n_test are the sample sizes of the training and testing data
# kappa is the estimated 4th moment of random error term
#############################
def get_dnn_p_val(mse_train, mse_test, n_train, n_test, kappa):
    test_stat = 1/np.sqrt(kappa*(1/n_train + 1/n_test))*(mse_train - mse_test)
    p_val = norm.sf(abs(test_stat)) * 2
    print(test_stat)
    print(p_val)
    return p_val

#-------------------------------------------------------------------------------------------------------------------#
###############################
# p-value combination
###############################
def p_val_combine(p_val, method='cauchy'):
    if method == 'hommel':
        U = len(p_val)
        q = np.arange(U) + 1
        p_order = np.sort(p_val)
        C_U = np.sum(1/q)

        print(f"U: {U}")
        print(f"q: {q}")
        print(f"p_order: {p_order}")
        print(f"C_U: {C_U}")


        p_val_combine = np.minimum(1, np.min(C_U * (U/q) * p_order))

        print(f"p_val_combine (hommel): {p_val_combine}")

    elif method == 'cauchy':
        T = np.mean(np.tan((0.5 - p_val) * pi))
        p_val_combine = cauchy.sf(T) 
    else:
        raise ValueError("Invalid Method!")
    return p_val_combine 

#-------------------------------------------------------------------------------------------------------------------#
def main(n_sample, tag = 5, nS = 5, gamma = 0.5, sigma = 0.5, max_hidden_unit = 16,
         deg = 1/3, optimizer = 'adam', epochs_nn = 1000, U = 5,
         epochs_dnn=1000, fit_test_data = True, batches = 10,
         early_stopping = True, validation_split = 0.2, 
         patience_nn = 5, patience_dnn = 10, min_delta = 1e-4, plus = True, drop_rate = 0.2,
         verbose = 0):
    sample_len = len(n_sample)
    p_val_lm = np.zeros((nS, sample_len))
    p_val_nn_hommel = np.zeros((nS, sample_len))
    p_val_dnn_hommel = np.zeros((nS, sample_len))
    p_val_dnn2_hommel = np.zeros((nS, sample_len))
    p_val_nn_cauchy = np.zeros((nS, sample_len))
    p_val_dnn_cauchy = np.zeros((nS, sample_len))
    p_val_dnn2_cauchy = np.zeros((nS, sample_len))
    p_nn = np.zeros(U)
    p_dnn = np.zeros(U)
    p_dnn2 = np.zeros(U)
    #l2_accuracy_nn = np.zeros((nS, sample_len))
    #l2_accuracy_dnn = np.zeros((nS, sample_len))
    #mse_nn = np.zeros((nS, sample_len))
    #mse_dnn = np.zeros((nS, sample_len))
    
    for s in range(nS):
        for i in range(len(n_sample)):
            nN = n_sample[i]
            n_train = np.floor(nN * gamma).astype(int)
            n_test  = nN - n_train
            X_mat, Y_true = generate_data(nN = nN)
            random_err = np.random.normal(loc = 0, scale = sigma, size = nN).reshape(Y_true.shape)
            Y = Y_true + random_err
            Y_centered = Y - np.mean(Y)
            
            batch_size = np.minimum(np.floor(n_train/batches).astype(int), np.floor(n_test/batches).astype(int)).astype(int)

            # Linear regression p-values
            X_tag = X_mat[tag,:].reshape((1, nN))
            lm_p = get_linear_reg_p_val(X_tag.T, Y.T)
            p_val_lm[s, i] = lm_p[1]

            for u in range(U):
                # Sample training and testing data
                train_id = np.random.choice(nN, size = n_train, replace = False)
                test_id = np.setdiff1d(range(nN), train_id)
                X_train  = X_tag[:,train_id]
                Y_train = Y_centered[:,train_id]
                Y_test  = Y[:,test_id] 
                
                mse_test_nn = np.mean(np.power(Y_test - np.mean(Y_test), 2))
                mse_test_dnn = np.mean(np.power(Y_test - np.mean(Y_test), 2))
                mse_test_dnn2 = np.mean(np.power(Y_test - np.mean(Y_test), 2))
                
                print(mse_test_nn)

                # Shallow ReLU network p-values
                n_h = np.floor(np.power(n_train, deg)).astype(int)
                mse_train_nn, predict_nn = fit_shallow_relu_nn(X_train, Y_train, n_h,
                                                               optimizer = optimizer, epochs = epochs_nn, 
                                                               batch_size = batch_size, 
                                                               early_stopping = early_stopping,
                                                               validation_split = validation_split,
                                                               patience = patience_nn,
                                                               min_delta = min_delta,
                                                               drop_rate = drop_rate,
                                                               verbose = verbose)
                kappa_nn = calculate_kappa(Y_train, predict_nn, plus = plus)
                # mse_nn[s,i] = mse_train_nn
                # l2_accuracy_nn[s,i] = np.mean(np.power(Y_true[:,train_id] - predict_nn.T, 2))
                p_nn[u] = get_dnn_p_val(mse_train_nn, mse_test_nn, n_train, n_test, kappa_nn)

                # Deep ReLU network p-values
                num_layers = np.floor(np.power(n_train, deg)).astype(int)
                num_nodes = np.repeat(max_hidden_unit, num_layers)
                mse_train_dnn, predict_dnn = fit_deep_relu_nn(X_train, Y_train, num_nodes, num_layers,
                                                               optimizer = optimizer, epochs = epochs_nn, 
                                                               batch_size = batch_size, 
                                                               early_stopping = early_stopping,
                                                               validation_split = validation_split,
                                                               patience = patience_nn,
                                                               min_delta = min_delta,
                                                               drop_rate = drop_rate,
                                                               verbose = verbose)
                kappa_dnn = calculate_kappa(Y_train, predict_dnn, plus = plus)

                # mse_dnn[s,i] = mse_train_dnn
                # l2_accuracy_dnn[s,i] = np.mean(np.power(Y_true[:,train_id] - predict_dnn.T, 2))
                p_dnn[u] = get_dnn_p_val(mse_train_dnn, mse_test_dnn, n_train, n_test, kappa_dnn)
                
                # Deep ReLU network 2 p-values
                num_layers2 = np.floor(np.power(n_train, (1/2)*deg)).astype(int)
                num_nodes2 = np.repeat(num_layers2, num_layers2)
                mse_train_dnn2, predict_dnn2 = fit_deep_relu_nn(X_train, Y_train, num_nodes, num_layers,
                                                               optimizer = optimizer, epochs = epochs_nn, 
                                                               batch_size = batch_size, 
                                                               early_stopping = early_stopping,
                                                               validation_split = validation_split,
                                                               patience = patience_nn,
                                                               min_delta = min_delta,
                                                               drop_rate = drop_rate,
                                                               verbose = verbose)
                kappa_dnn2 = calculate_kappa(Y_train, predict_dnn2, plus = plus)
                # mse_dnn[s,i] = mse_train_dnn
                # l2_accuracy_dnn[s,i] = np.mean(np.power(Y_true[:,train_id] - predict_dnn.T, 2))
                p_dnn2[u] = get_dnn_p_val(mse_train_dnn2, mse_test_dnn2, n_train, n_test, kappa_dnn2)
                
            print(p_nn)
            print(p_dnn)
            print(p_dnn2)
            p_val_nn_hommel[s,i] = p_val_combine(p_nn, method = 'hommel')
            p_val_nn_cauchy[s,i] = p_val_combine(p_nn, method = 'cauchy')
            p_val_dnn_hommel[s,i] = p_val_combine(p_dnn, method = 'hommel')
            p_val_dnn_cauchy[s,i] = p_val_combine(p_dnn, method = 'cauchy')
            p_val_dnn2_hommel[s,i] = p_val_combine(p_dnn2, method = 'hommel')
            p_val_dnn2_cauchy[s,i] = p_val_combine(p_dnn2, method = 'cauchy')


    return p_val_lm, p_val_nn_hommel, p_val_nn_cauchy, p_val_dnn_hommel, p_val_dnn_cauchy, p_val_dnn2_hommel, p_val_dnn2_cauchy

#-------------------------------------------------------------------------------------------------------------------#
def print_array(arr):
    """
    prints a 2-D numpy array in a nicer format
    """
    for a in arr:
        for elem in a:
            print("{}".format(elem).rjust(3), end="\t")
        print(end="\n")
        
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
