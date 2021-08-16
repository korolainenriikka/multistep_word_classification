# Multistep workflow: word classification

This repository contains a multistep MLflow pipeline that does the following steps:

* loads lists of Finnish and English words into the MLflow artifact store
* transforms the words into letter frequency arrays
* trains a word classifier with this data
    * the classifier uses 5-fold cross-validation and logs all achieved accuracy scores as metrics (accuracy0/1/2/3 in metrics)

The code used was originally a solution to (a ml course exercise)[https://csmastersuh.github.io/data_analysis_with_python_summer_2021/bayes.html#Exercise-3-(word-classification)], part of the code is copied from the exercise template.

notes on the versions:
 * branch conda-version has a working configuration using a conda environment
 * main issue is that MLflow tries to create a new container for each step and this leads to nested containers.
    * current Dockerfile installs docker inside the container
    * the volume in MLProject binds word_classifier containers' docker socket to the outer machines' docker socket. Like this the step containers should not be nested but siblings to the workflow container
    * current error in run command despite this: run not found.

