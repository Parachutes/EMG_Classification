# EMG_Classification

This repository contains the source code and datasets used in the Shichao Ma's individual project.

The original dataset is from EMG DATASETS REPOSITORY (https://www.rami-khushaba.com/electromyogram-emg-repository.html) and all the samples are contributed by the owner of the repository and his colleagues. (R. N. Khushaba and Sarath Kodagoda, “Electromyogram (EMG) Feature Reduction Using Mutual Components Analysis for Multifunction Prosthetic Fingers Control”‏, in Proc. Int. Conf. on Control, Automation, Robotics & Vision (ICARCV), Guangzhou, 2012, pp. 1534-1539. (6 pages)) 

All the source code in repository is written in Python and developed by Shichao Ma.

To run the code make sure that you have at least installed Python 3 (3.5 or 3.6 is preferable) and TensorFlow (a GPU version is preferable, the CPU version also works but it usually take ages to finish the classification. If you are using the GPU version then make sure that relevant drivers such as the NIVIDIA driver have been installed as well) in your computer.

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
  
