# EMG_Classification

This repository contains the source code and datasets used in the Shichao Ma's individual project.

Currently there is only the original dataset (data/raw) which contains the raw EMG samples included in the repository. The original dataset is from EMG DATASETS REPOSITORY (https://www.rami-khushaba.com/electromyogram-emg-repository.html) and all the samples are contributed by the owner of the repository and his colleagues. (R. N. Khushaba and Sarath Kodagoda, “Electromyogram (EMG) Feature Reduction Using Mutual Components Analysis for Multifunction Prosthetic Fingers Control”‏, in Proc. Int. Conf. on Control, Automation, Robotics & Vision (ICARCV), Guangzhou, 2012, pp. 1534-1539. (6 pages)) 

All the source code in repository is written in Python and developed by Shichao Ma.

To run the code make sure that you have at least installed Python 3 (3.5 or 3.6 is preferable) and TensorFlow (a GPU version is strongly recommanded, the CPU version also works but it usually take ages to finish the classification of convolutional neural network. If you are using the GPU version then make sure that relevant drivers such as the NIVIDIA driver have been installed as well) in your computer.

There are some external libraries imported in the source code so you need to install following packages:

  0). pip3
  
    So that other packages can be easily installed. If you are using Ubuntu 16.04.6 or other Linux as your operating system you can simply install it by executing "sudo apt install python3-pip" in your command line ( following commands are also executed in Ubuntu 16.04.6).

  1). PyWavelets
  
    You can install it by executing "pip3 install PyWavelets" in your command line.
    
  2). scipy
  
    You can install it by executing "pip3 install scipy" in your command line.
    
  3). numpy
  
    You can install it by executing "pip3 install numpy" in your command line.
    
  4). keras
  
    You can install it by executing "pip3 install keras" in your command line.    
    
  5). sklearn
  
    You can install it by executing "pip3 install sklearn" in your command line. 
  
  
  
  Once all these packages have been successfully installed and this repository has been cloned to the local, you can start to produce Dataset_A (data/features_windowing) and Dataset_B (data/raw_windowing) (for more details about Dataset_A and Dataset_B please check the report) by simply executing "DataModification.py" under the directory "/src". 
  
    python3 DataModification.py
    
This process usually takes around 15 minutes and the size of combination of Dataset_A and Dataset_B around 4 GB. After two new datasets are produced, you can then do the classification by simply executing relevant classifier files under the directory "/src".

    python3 ClassifierKNN.py
    
    python3 ClassifierSVM.py
    
    python3 ClassifierNN.py
    
All of these three classifiers will classify the samples in Dataset_A.

    python3 ClassifierCNN.py
    
Convolutonal neural network will classify the samples in Dataset_B.

The accuracies will be given in the end of execution of each classifier.

If you want to testify the robustness of each classifer, you can execute the "write_noisy_raw_windowing(crop_size, window_size, interval, sigma)" and "write_noisy_features_windowing(crop_size, window_size, interval, sigma)" these two functions in the DataModification.py and two more noisy dataset (data/features_windowing_noisy and data/raw_windowing_noisy ) will be produced. After that, you can change the directory of test dataset in corresponding classifier file to get the accuracy on the noisy dataset.

You can also freely select different samples as training samples and test samples by simply modifying some parameters in classifer files. In the report the first two trials contributed by each subject are selected as the training samples and the last trials are selected as the test sample.

PS. Please be patient when you are doing the classification on the CNN classifier, it takes around 10 minutes to collect data from csv files and the training takes around 20 minutes if you are using powerful GPUs (such as NVIDIA Tesla K80 offered by Google Cloud Platform). However, the training may take several days if you are using CPUs :-(





  
