#Shichao.Ma$$$IndividualProject
#The KNN classifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
from pathlib import Path
import Utility

class ClassifierKNN:

    data_training = []
    label_training = []
    data_testing = []
    predictions = []

    def __init__(self, data_training, label_training, data_testing):
        self.data_training = [d.flatten() for d in data_training]
        self.label_training = label_training
        self.data_testing = data_testing

    def get_predictions(self):     
        
        print(len(self.data_training[0]))
        i = 1
        
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(self.data_training, self.label_training)
        for d_t in self.data_testing:
            d_t = [d.flatten() for d in d_t]
            d_t = np.array(d_t).reshape(len(d_t), len(d_t[0]))
            prediction = neigh.predict(d_t)
            prediction = prediction.tolist()
            
            
            print(prediction)
            print(i)
            i = i + 1
            
            self.predictions.append(max(set(prediction), key=prediction.count))
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

classifierKNN = ClassifierKNN(x_train, y_train, x_test)
print("The KNN Accuracy: ", Utility.get_accuracy(classifierKNN.get_predictions(), y_test))
