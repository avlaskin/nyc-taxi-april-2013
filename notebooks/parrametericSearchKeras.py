from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import time
import numpy as np
from cleanData import getPredictionFeatures, getCleanMergedData, getTripData, getTrainingTestSet, getMonolithTrainingSet
from sklearn.metrics import *

def getKerasModelOfParams(params, input_dim):
    ''' Creates a NN based on parameters. Support up to 2 hidden layers.'''
    model = Sequential()
    print("Params {} Input_dim {}".format(params, input_dim))
    model.add(Dense(params[0], input_dim=input_dim, activation='relu', kernel_initializer="normal"))
    if len(params) > 1:
        model.add(Dense(params[1], activation='relu', kernel_initializer="normal"))
    if len(params) > 2:
        model.add(Dense(params[2], activation='relu', kernel_initializer="normal"))
    model.add(Dense(1, activation='relu', kernel_initializer="normal")) #uniform
    model.compile(loss=losses.mean_squared_error, optimizer='sgd', metrics=['mae'])
    return model

def trainKerasModel(model, train, train_target, idx=0, epochs=30, mono=False):
    ''' Trains model according to parameters. Good number of epochs are more than 30.'''
    bestFile = "best_%d" % idx
    mcp = ModelCheckpoint(bestFile, monitor="loss", save_best_only=True, save_weights_only=False)
    if mono:
        model.fit(train.values, train_target, validation_split=0.3, epochs=epochs, batch_size=6, verbose=0, callbacks=[mcp])
    else:
        model.fit(train.values, train_target, epochs=epochs, batch_size=6, verbose=0, callbacks=[mcp])
    # Keras does not keep best model unless we save it with callback and load again here
    model.load_weights(bestFile)
    return model

def getKerasModelScore(model, test, test_target):
    '''We score models based on R2, variance explained and mean square error. Returns back scores as an array.'''
    keras_predicted = model.predict(test.values)
    keras_predicted_cv = model.predict(test.values)
    variance_explained = explained_variance_score(test_target, keras_predicted)
    r2 = r2_score(test_target, keras_predicted)
    mse = mean_squared_error(test_target, keras_predicted)
    return [variance_explained, r2, mse]

def creatLearningCurve(input_dim, data, data_target, test, test_target):
    ''' Gradually feeding data to the best model we can construct and analisy an learning curve for NN.'''
    percents = [20, 30, 40, 50, 60, 70, 80, 90, 99]
    testLoss = []
    trainLoss = []
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    for i in range(0, len(percents)):
        for p in percents:
            number_of_samples = int((p*len(data_target)) / 100)
            # x_train = data[:number_of_samples]
            # y_train = data_target[:number_of_samples]
            x_train = data
            y_train = data_target
            model = Sequential()
            model.add(Dense(17, input_dim=input_dim, activation='relu', kernel_initializer="normal"))
            model.add(Dense(1, activation='relu', kernel_initializer="normal"))
            model.compile(loss=losses.mean_squared_error, optimizer='sgd', metrics=['mae'])
            m = trainKerasModel(model, x_train, y_train, idx=29, epochs=3)
            train_score = m.evaluate(x_train, y_train, verbose=0)
            test_score = m.evaluate(test, test_target, verbose=0)
            testLoss.append(test_score)
            trainLoss.append(train_score)
            print("Indexes: {} Train {} Test {}".format(number_of_samples, train_score, test_score))
            tf.reset_default_graph() # Retraining sometimes fails, we reset tensorflow internals this way
    print("Percent: {}".format(percents))
    print("TEST: {}".format(testLoss))
    print("TRAIN: {}".format(trainLoss))

if __name__ == "__main__":
    print("Loading...", end='')
    s = time.time()
    data = getTripData('./data/loc_hash_3_trip_4_2013.csv')
    e = time.time()
    print("{}".format(e-s))
    print("Cleaning...", end='')
    s = time.time()
    data = getCleanMergedData(data)
    e = time.time()
    print("{}".format(e-s))
    print("Making features...", end='')
    predictors = getPredictionFeatures(data)
    s = time.time()
    print("{}".format(s-e))
    # tip_amount tip_percent fare_amount
    predictors = predictors.drop(['tip_amount', 'tip_percent', 'is_card_payment'], axis=1)
    print("Training Predictors: {}".format(predictors.columns.values))
    print("Making training sets...", end='')
    s = time.time()
    sets = getTrainingTestSet(predictors, 'fare_amount', target_scaler=70.0)
    monoset = getMonolithTrainingSet(predictors, 'fare_amount', target_scaler=70.0)
    e = time.time()
    print("{}".format(e-s))
    train = sets[0][0]
    train_target = sets[0][1]
    test = sets[1][0]
    test_target = sets[1][1]
    cross_val = sets[2][0]
    cross_val_target = sets[2][1]
    # We add bais term as for linear regression model
    train['bais'] = 1.0
    test['bais'] = 1.0
    cross_val['bais'] = 1.0
    print("Predicting {}".format(train_target[:10]))
    print("Training NN Predictors: {}".format(train.columns.values))
    # TESTED PARAMS - best so far is [18]
    #params = [[10], [17], [34], [8, 17], [17, 20], [17, 30], [17, 20, 10]]
    #params = [[10], [13], [14], [15], [16], [17], [18], [19], [17, 5], [5, 17],[3, 20], [17,2], [17, 8, 2]]
    params = [[10], [14], [16], [17], [18], [18, 2]]
    score = []
    models = []
    i = 0
    dim = len(train.columns.values)
    for p in params:
        m = getKerasModelOfParams(p, dim)
        m = trainKerasModel(m, train, train_target, i, epochs=30)
        scores = getKerasModelScore(m, test, test_target)
        print(scores)
        variance_exp = scores[0]
        r2score = scores[1]
        mseScore = scores[2]
        score.append(variance_exp)
        print("Model {} Score = {} R2.score = {} MSE = {}".format(i, variance_exp, r2score, mseScore))
        i += 1
    print(score)
    print("Best model: {}".format(np.argmax(score)))
    print("Best params: {}".format(params[np.argmax(score)]))
    ##############################################
    # This code is not working with Tensorflow 1.6 as it does not reset the tf session properly.
    # So that it crashes when we try to train a new model several times.
    #creatLearningCurve(dim, train, train_target, test, test_target)
