#Shichao.Ma$$$IndividualProject
#This is the Conventional Neural Network
import numpy as np
import tensorflow as tf
from tensorflow import keras
import Utility
import random as rn


#To avoid the randomness
import os
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)
rn.seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)






class ClassifierCNN1D:

    data_training = []
    label_training = []
    data_testing = []

    predictions = []

    init_weights = keras.initializers.glorot_normal(seed=1);
    regularizer = keras.regularizers.l2(l=0.00001)



    def __init__(self, data_training, label_training, data_testing):
        self.data_training = np.array(data_training).reshape(len(data_training), 3000, 1)
        self.label_training = [Utility.label_str2array(l) for l in label_training]
        self.label_training = np.array(self.label_training).reshape(len(label_training), 15)

        self.data_testing = data_testing




    def train_the_model(self):
        model = keras.models.Sequential()
        # TODO more convolutional layers, extract more useful features
        model.add(keras.layers.Conv1D(filters=100, kernel_size=50, activation='tanh', input_shape=(3000, 1), padding='same'))
        model.add(keras.layers.MaxPooling1D(pool_size=5))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Conv1D(filters=50, kernel_size=30, activation='tanh', kernel_initializer=self.init_weights, padding='same'))
        model.add(keras.layers.MaxPooling1D(pool_size=3))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Conv1D(filters=30, kernel_size=20, activation='tanh', kernel_initializer=self.init_weights, padding='same'))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(150, activation='tanh'))
        model.add(keras.layers.Dense(120, activation='tanh'))
        #model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(15, activation=tf.nn.softmax))
        opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['accuracy'])
        early_stopping = keras.callbacks.EarlyStopping(monitor='acc', patience=10, verbose=0,
                                                       mode='auto', baseline=None)
        model.fit(self.data_training,
                  self.label_training,
                  epochs=100,
                  batch_size=40,
                  shuffle=True,
                  callbacks=[early_stopping])







        # Do the prediction
        for d_t in self.data_testing:
            d_t = np.array(d_t).reshape(len(d_t), 3000, 1)
            prediction = model.predict(d_t)
            prediction = [Utility.label_num2str(np.argmax(p)) for p in prediction]
            self.predictions.append(max(set(prediction), key=prediction.count))


    #To get the prediction through the model
    def get_predictions(self):
        return self.predictions








