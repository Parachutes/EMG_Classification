
#DONE
#Shichao.Ma$$$IndividualProject
#This is the Conventional Neural Network
import numpy as np
import tensorflow as tf
from tensorflow import keras
import Utility
import random as rn
from pathlib import Path
import os

#To avoid some randomness
import os
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)
rn.seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)



class ClassifierCNN:
    
    data_training = []
    label_training = []
    data_testing = []
    
    predictions = []
    
    regularizer = keras.regularizers.l2(l=0.00012)
    
    
    
    def __init__(self, data_training, label_training, data_testing):
        

        
        
        for d in data_training:
            matrix_3D = []
            for r in d:
                matrix_2D = np.array(r).reshape(100,100,1)
                matrix_3D.append(matrix_2D)
            self.data_training.append(matrix_3D)
            
        self.data_training = np.array(self.data_training)
        
        
        self.label_training = [Utility.label_str2array(l) for l in label_training]
        self.label_training = np.array(self.label_training).reshape(len(label_training), 15)
        
        
        self.data_testing = data_testing
    
    
    
    
    def train_the_model(self):
        model = keras.models.Sequential()



        model.add(keras.layers.Conv3D(filters=5, kernel_size=(3,5,5), activation=keras.layers.LeakyReLU(alpha=0.3), input_shape=(8,100,100,1), kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer))
        
        model.add(keras.layers.MaxPooling3D(pool_size=(1,2,2)))
        
        model.add(keras.layers.Conv3D(filters=5, kernel_size=(3,3,3), activation=keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer))
        
        model.add(keras.layers.MaxPooling3D(pool_size=(1,2,2)))
        
        model.add(keras.layers.Conv3D(filters=3, kernel_size=(3,4,4), activation=keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer))
        
        model.add(keras.layers.MaxPooling3D(pool_size=(2,2,2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(200, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer=self.regularizer, activity_regularizer=self.regularizer))
        model.add(keras.layers.Dense(150, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer=self.regularizer, activity_regularizer=self.regularizer))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(15, activation=tf.nn.softmax))

        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
            
        early_stopping = keras.callbacks.EarlyStopping(monitor='acc', patience=10, verbose=0, mode='auto', baseline=None)

        model.fit(self.data_training,self.label_training, epochs=800, batch_size=20, shuffle=True, callbacks=[early_stopping])
                      
                      
                      
        # Do the prediction
        for d_t in self.data_testing:
            
            matrix_3D_testset = []
            for d in d_t:
                matrix_3D = []
                for r in d:
                    matrix_2D = np.array(r).reshape(100,100,1)
                    matrix_3D.append(matrix_2D)
                matrix_3D_testset.append(matrix_3D)

            prediction = model.predict(np.array(matrix_3D_testset))
            prediction = [Utility.label_num2str(np.argmax(p)) for p in prediction]
            self.predictions.append(max(set(prediction), key=prediction.count))


    #To get the prediction through the model
    def get_predictions(self):
        return self.predictions




#Script part
current_path = os.path.abspath(os.path.dirname(__file__))
current_path_parent = str(Path(current_path).parent)
path_dataset = current_path_parent + '/data/raw_windowing'

x_train = []
y_train = []
x_test = []
y_test = []

Utility.collect_data_with_windowing(path_dataset, x_train, y_train, ["S1", "S2"], ["1","2"])
Utility.collect_testing_data_with_windowing(path_dataset, x_test, y_test, ["S1", "S2"], ["3"])

print("Hello World")

classifierCNN = ClassifierCNN(x_train, y_train, x_test)

print(np.array(classifierCNN.data_training).shape)

classifierCNN.train_the_model()
print("The CNN Accuracy: ", Utility.get_accuracy(classifierCNN.get_predictions(), y_test))

