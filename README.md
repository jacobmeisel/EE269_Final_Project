# Classification of Cardiovascular Arrythmia using ECG Signals

Approximately 5% of the US population develops an arrhythmia every year.
A University of Pennsylvania study suggests that COVID-19 positive patients are 10 times more likely to develop an arrhythmia than those without.
In this project, we aim to accurately classify normal (N) heartbeats from four different types of arrhythmias:
1) Supraventricular premature beat (S)
2) Premature ventricular contraction (V)
3) Fusion of ventricular and normal beat (F)
4) Unclassifiable beat (Q)


We used several models including:
1) naive bayes
2) GMM (with PCA)
3) KNN
4) SVM (with PCA)
5) Modified pre-trained GoogleNet
6) Custom CNN


We used several features including:
1) Raw ECG Waveform
2) DFT Magnitude
3) Discrete Wavelet Transform (DWT)
   i) db4 wavelet
   ii) sym5 wavelet
4) Continuous Wavelet Transform (CWT)
   i) 224 x 224 x 3 images
   ii) 32 x 32 x 3 images
   iii) 32 x 32 x 3 images using rotation, scaling, and reflection for data augmentation.


See the project poster, presentation, and report for more details.
