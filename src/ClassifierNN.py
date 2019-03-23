#Shichao.Ma$$$IndividualProject
#This is the Traditional Neural Network Classifier


import numpy as np
import tensorflow as tf
import Utility
from tensorflow import keras
import random as rn
import os
from pathlib import Path







class ClassifierNN:

    #Fields
    data_training = []
    label_training = []
    data_testing = []
    predictions = []
    input_size = 0


    regularizer = keras.regularizers.l2(l = 0.00001)

    #The constructor
    def __init__(self, data_training, label_training, data_testing):

        self.data_training = [d.flatten() for d in data_training]
        self.input_size = len(self.data_training[0])
        self.data_training = np.array(self.data_training).reshape(len(data_training), self.input_size)
        self.label_training = [Utility.label_str2array(l) for l in label_training]
        self.label_training = np.array(self.label_training).reshape(len(label_training), 15)
        self.data_testing = data_testing
        


    #This method is to train the nn model
    def train_the_model(self):
        model = keras.models.Sequential()
        #The layers of NN
        model.add(keras.layers.Dense(200, activation='relu', input_dim=self.input_size,
                                     kernel_regularizer = self.regularizer))
        #model.add(keras.layers.Dropout(0.6))
        model.add(keras.layers.Dense(15, activation=tf.nn.softmax,
                                     kernel_regularizer = self.regularizer))
        #Optimizers
        opt = keras.optimizers.Adam(lr=0.0007)
        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['accuracy'])
        early_stopping = keras.callbacks.EarlyStopping(monitor='acc', patience=20, verbose=0, mode='auto', baseline=None)
        model.fit(self.data_training, self.label_training,
                  epochs=2000,
                  batch_size=30,
                  callbacks=[early_stopping],
                  shuffle=True)


        # Do the prediction
        for d_t in self.data_testing:
            d_t = [d.flatten() for d in d_t]
            d_t = np.array(d_t).reshape(len(d_t), self.input_size)
            prediction = model.predict(d_t)
            prediction = [Utility.label_num2str(np.argmax(p)) for p in prediction]
            self.predictions.append(max(set(prediction), key=prediction.count))



    #To get the prediction through the model
    def get_predictions(self):
        return self.predictions



#Script part
current_path = os.path.abspath(os.path.dirname(__file__))
current_path_parent = str(Path(current_path).parent)
path_dataset = current_path_parent + '/data/features_windowing'

x_train = []
y_train = []
x_test = []
y_test = []

Utility.collect_data_with_windowing(path_dataset, x_train, y_train, ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"], ["1", "2"])
Utility.collect_testing_data_with_windowing(path_dataset, x_test, y_test, ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"], ["3"])


classifierNN = ClassifierNN(x_train, y_train, x_test)
classifierNN.train_the_model()
result = Utility.get_accuracy(classifierNN.get_predictions(), y_test)
print("The NN Accuracy: ",result)


