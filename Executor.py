import Utility
import os
import csv
from ClassifierCNN1D import ClassifierCNN1D
import numpy as np
from tensorflow import keras


current_path = os.path.abspath(os.path.dirname(__file__))
directory_cnn_windowing = current_path + '/Data_CNN_Windowing'


data_training_window_cnn = []
labels_training_window_cnn = []
data_testing_window_cnn = []
labels_testing_window_cnn = []


Utility.collect_data_with_windowing(directory_cnn_windowing, data_training_window_cnn, labels_training_window_cnn, ["S1", "S2", "S3","S4", "S5", "S6","S7", "S8"], ["1","2"])
#Utility.collect_testing_data_with_windowing(directory_cnn_windowing, data_testing_window_cnn, labels_testing_window_cnn, ["S1", "S2", "S3","S4", "S5", "S6","S7", "S8"], ["3"])
Utility.collect_data_with_windowing(directory_cnn_windowing, data_testing_window_cnn, labels_testing_window_cnn, ["S1", "S2", "S3","S4", "S5", "S6","S7", "S8"], ["3"])

classifierCNN1D = ClassifierCNN1D(data_training_window_cnn, labels_training_window_cnn, data_testing_window_cnn)
classifierCNN1D.train_the_model()
print("The CNN1D Accuracy: ", Utility.get_accuracy(classifierCNN1D.get_predictions(), labels_testing_window_cnn))












