#Shichao.Ma$$$IndividualProject
#This is the Conventional Neural Network
import numpy as np
import tensorflow as tf
from tensorflow import keras
import Utility
import random as rn
from pathlib import Path
import os







class ClassifierCNN:

    data_training = []
    label_training = []
    data_testing = []
    label_testing = []
    
    predictions = []


    #Constructor
    def __init__(self, data_training, label_training, data_testing, label_testing):

        self.data_training = np.array(data_training).reshape(np.array(data_training).shape + (1,))
        self.label_training = [Utility.label_str2array(l) for l in label_training]
        self.label_training = np.array(self.label_training).reshape(len(label_training), 15)
        
        self.data_testing = np.array(data_testing).reshape(np.array(data_testing).shape + (1,))
        self.label_testing = [Utility.label_str2array(l) for l in label_testing]
        self.label_training = np.array(self.label_testing).reshape(len(label_testing), 15)
    
    

    #Train the neural network
    def train_the_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters = 15, kernel_size=(8,50), activation='tanh', input_shape=self.data_training[0].shape, padding='same', kernel_regularizer=self.regularizer))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 5)))
        #model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Conv2D(filters = 10, kernel_size=(4,30), activation='tanh', padding='same', kernel_regularizer=self.regularizer))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        #model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Conv2D(filters = 5, kernel_size=(2,10), activation='tanh', padding='same', kernel_regularizer=self.regularizer))
        model.add(keras.layers.MaxPooling2D(pool_size=(1, 2)))
        #model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(200, activation='tanh', kernel_regularizer=self.regularizer))
        #model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(150, activation='tanh', kernel_regularizer=self.regularizer))
        #model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(15, activation=tf.nn.softmax, kernel_regularizer=self.regularizer))

        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['accuracy'])
        
        #early_stopping = keras.callbacks.EarlyStopping(monitor='acc', patience=8, verbose=0, mode='auto', baseline=None)
        model.fit(self.data_training,
                  self.label_training,
                  epochs=800,
                  batch_size=30,
                  shuffle=True,
                  #callbacks=[early_stopping]
                  validation_data=(self.data_testing, self.label_testing))




    #To get the prediction through the model
    #def get_predictions(self):
        #return self.predictions




#Script part
current_path = os.path.abspath(os.path.dirname(__file__))
current_path_parent = str(Path(current_path).parent)
path_dataset = current_path_parent + '/data/raw_windowing'

x_train = []
y_train = []
x_test = []
y_test = []

Utility.collect_data_with_windowing(path_dataset, x_train, y_train, ["S1", "S2", "S3","S4", "S5", "S6","S7", "S8"], ["1","2"])
Utility.collect_data_with_windowing(path_dataset, x_test, y_test, ["S1", "S2", "S3","S4", "S5", "S6","S7", "S8"], ["3"])


classifierCNN = ClassifierCNN(x_train, y_train, x_test, y_test)
classifierCNN.train_the_model()
#print("The CNN Accuracy: ", Utility.get_accuracy(classifierCNN.get_predictions(), y_test))




