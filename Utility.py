import numpy as np
import os
import csv
import re




#$$$$ functions for data handling
def read_csv(f):
    matrix = []
    #TODO crop???
    i = 0
    with f as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            row = [float(i) for i in row]
            matrix.append(row)
            i = i + 1
            if i >= 70000:
                break
    csvFile.close()
    return matrix




#TODO another collection function/ collect data per channel!
def collect_data_with_windowing(directory, data, labels, subject, index):
    for filename in os.listdir(directory):
        for s in subject:
            for i in index:
                x = re.findall(s + "_..." + i + "_" + "\d\d", filename)
                if x:
                    os.chdir(directory)
                    f = open(filename)
                    matrix = np.array(read_csv(f))
                    label = os.path.splitext(filename)[0][3:6]
                    data.append(matrix)
                    labels.append(label)

def collect_testing_data_with_windowing(directory, data, labels, subject, index):
    for s in subject:
        for m in ["HC_", "I_I", "I_M", "IMR", "L_L", "M_M", "M_R", "MRL", "R_L", "R_R", "T_I", "T_L", "T_M", "T_R", "T_T"]:
            for i in index:
                combination_matrix = []
                for filename in os.listdir(directory):
                    x = re.findall(s + "_" + m + i + "_" + "\d\d", filename)
                    if x:
                        os.chdir(directory)
                        f = open(filename)
                        matrix = np.array(read_csv(f))
                        combination_matrix.append(matrix)
                data.append(combination_matrix)
                labels.append(m)






#$$$$ functions for classification
def get_accuracy(predictions, labels_test):
    accuracy = 0
    for i in range(len(labels_test)):
        if predictions[i] == labels_test[i]:
            accuracy = accuracy + 1
    return accuracy/len(labels_test)

def label_str2num(label):
    return {
        'HC_': 0,
        'I_I': 1,
        'I_M': 2,
        'IMR': 3,
        'L_L': 4,
        'M_M': 5,
        'M_R': 6,
        'MRL': 7,
        'R_L': 8,
        'R_R': 9,
        'T_I': 10,
        'T_L': 11,
        'T_M': 12,
        'T_R': 13,
        'T_T': 14,
    }[label]

def label_str2array(label):
    return {
        'HC_': [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        'I_I': [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        'I_M': [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        'IMR': [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        'L_L': [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        'M_M': [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        'M_R': [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        'MRL': [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
        'R_L': [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
        'R_R': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
        'T_I': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
        'T_L': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
        'T_M': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
        'T_R': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
        'T_T': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    }[label]

def label_num2str(label):
    return {
        0: 'HC_',
        1: 'I_I',
        2: 'I_M',
        3: 'IMR',
        4: 'L_L',
        5: 'M_M',
        6: 'M_R',
        7: 'MRL',
        8: 'R_L',
        9: 'R_R',
        10: 'T_I',
        11: 'T_L',
        12: 'T_M',
        13: 'T_R',
        14: 'T_T',
    }[label]


