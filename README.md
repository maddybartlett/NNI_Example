# NNI_Example

This repository contains the slides for a presentation introducing NNI, and a set of scripts which can be used as an exercise to learn how to create an NNI hyperparameter tuning experiment. 

NNI website: https://nni.readthedocs.io/en/latest/Tutorial/QuickStart.html
NNI github: https://github.com/microsoft/nni/


## Exercise

In the exercise directory, a number of scripts are provided which allow you to practice setting up an NNI experiment for a simple neural network. 

simple_NN.ipynb trains and tests a simple neural network with 2 sets of hyperparameters. Make sure you can run this script first.

The network has also been saved in net.py, and the data loader in data_fetch.py. 
A trial script has been written in experiment_trial.py and there is an empty YAML file (config.yml).

Create an NNI experiment by following these steps:
1.	Edit the trial script so that the hyperparameters are fetched from NNI, and the results are reported to NNI
2.	In the YAML file, define your search space and configure your experiment. Hint: copy and paste the examples from the NNI website and edit them to suit your experiment.
3.	Run your experiment from the command line (you might need to do this from an Anaconda Prompt) with the command: nnictl create --config config.yml

