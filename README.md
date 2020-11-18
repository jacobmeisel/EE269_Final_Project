# Classification of Cardiovascular Arrythmia using ECG Signals

## Introduction
Approximately 5% of the US population develops an arrhythmia every year.
A University of Pennsylvania study suggests that COVID-19 positive patients are 10 times more likely to develop an arrhythmia than those without.
In this project, we aim to accurately classify normal (N) heartbeats from four different types of arrhythmias:
1) Supraventricular premature beat (S)
2) Premature ventricular contraction (V)
3) Fusion of ventricular and normal beat (F)
4) Unclassifiable beat (Q)

## Data Set
We used the [MIT-BIH arrhythmia database](https://www.kaggle.com/shayanfazeli/heartbeat?select=mitbih_train.csv).

* Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification: A Deep Transferable Representation." arXiv preprint arXiv:1805.00794 (2018).

## Features
We used several features including:
1) Raw ECG Waveform
2) DFT Magnitude
3) Discrete Wavelet Transform (DWT):
   1. db4 wavelet
   2. sym5 wavelet
4) Continuous Wavelet Transform (CWT):
   1. 224 x 224 x 3 images
   2. 32 x 32 x 3 images
   3. 32 x 32 x 3 images using rotation, scaling, and reflection for data augmentation.

## Models
We used several models including:
1) Naive Bayes
2) GMM (with PCA)
3) KNN
4) SVM (with PCA)
5) Modified pre-trained GoogleNet
6) Custom CNN

## Results
The best classifier was KNN (k=3) with sym5 DWT features.
The top classifiers were KNN and cubic SVM (with PCA) for raw waveform and DWT features.
See the project [poster](https://drive.google.com/file/d/1pKp4gdpWvOgbXFXwuuydGk7LyDGKqDk0/view?usp=sharing), [presentation](https://drive.google.com/file/d/1oLLiuGJkwxZy312y3RXbC9OqykiouUYC/view?usp=sharing), and [report]() for more results and conclusions.
