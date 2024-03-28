# To fall or not to fall

In the project we tried to apply all our knowledge in the area of statisitcal learning by focusing on:
- feature extraction through signal processing techniques
- training of various classification models (SVM, KNN, Logistic, NN etc.)
- hyperparamenter selection for each model and model selection
- compression of deep learning models

We invite you to read `REPORT.pdf` for a more detailed explanation of the algorithms used and the reasoning followed.

The project has the following goals:
- to simulate a collection of falls by using specific sensors and experimenting with different
fall modes
- to try to mitigate the effect of bias due to the synthetic (simulation) approach by special
feature extraction algorithms using a small amount of real falls data.
- to develop fall classification algorithms both based on classical machine learning models
and neural network models
- to use the best model obtained to implement a practical solution to fall detection, based
on the TinyML framework.

## Project Structure

```plaintext
ToFall_orNot_toFall-main
│   .gitignore                
│   LICENSE                   
│   README.md                 # The file you are reading right now
│   REPORT.pdf                # Detailed project report
│   desktop.ini               # Windows folder settings file
│   main.ipynb                # Jupyter notebook with the project's main code, using only classic ML algorithms (both supervied and unsup. frameworks)
│
├───data collection           # Scripts and instructions for data collection
│       Accell&Gyrosc.ino     # Arduino sketch for accelerometer and gyroscope signals collection 
│       README.md             # Instructions for data collection
│       collect.py            # Python script to collect data from Arduino 
│
├───deep learning             # Deep learning models and scripts
│   │   deep_learning.ipynb   # Notebook with deep learning model training and evaluation
│   │
│   ├───models                # TensorFlow Lite model (compressed version of best TF model)
│   │   │   model.tflite
│   │   │   model_no_quant.tflite
│   │   │
│   │   └───model             # Best TensorFlow model weights directory
│   │       │   keras_metadata.pb
│   │       │   saved_model.pb
│   │       │
│   │       └───variables     # Model variables
│   │               variables.data-00000-of-00001
│   │               variables.index
│   │
│   └───tinyML                # TinyML model and scaling scripts for microcontrollers
│           StandardScaler.cc # C++ standard scaler implementation
│           StandardScaler.h 
│           main.ino          # Arduino main script for deploying the TinyML model
│           model.h           
│
├───feature extraction        # Scripts for extracting features from the dataset
│       feature_extraction.py # Python script for feature extraction
│       file_csv.zip          # Zipped CSV files with real falls to extract interesting features
│
├───images                    # Various images used in documentation and reports
│       ...
│
└───main functions            # Helper functions for the project
        nice_plots.py         # Functions for generating nice plots
        supervised_funcs.py   # Supervised learning helper functions used in the main notebook
```

## Data collection 

Refer to the `README.md` file in the `data collection` directory of this repository for specific info about the collection precess. We used accelerometer and gyroscope sensors to collect falls and normal activities signals simulating real-life context.

Reverse fall          |  Step | Fall
:-------------------------:|:-------------------------:|:-------------------------:
![rfall](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/d9514377-6b91-4484-adb2-71ac81f89bc1)|![step](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/705c2bdf-9bc4-4640-9ff5-aacc5e041316)|![fall](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/88cf4140-7ba4-4709-9c9e-d0275aedc57a)

## Features extraction

You can find the code and additional data for feature extraction in the `feature extraction` directory.

Fourier analysis of signals.

| <!-- -->    | <!-- -->    | 
|-------------|-------------|
![fft_gyr](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/a8dd17af-f599-437f-99ce-c4466e735a99)|![fft_acc](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/91ed2c13-b774-4337-9ff0-88d887785791)

Finding the most informative frequencies and applying a lowpass filter (we simply thresholded at the first k frequencies since the lower frequencies contain higher energy).

| <!-- -->    | <!-- -->    | 
|-------------|-------------|
![maxbin_gyr](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/52c1e962-a924-44ad-8d6e-8dd3ea489b1a)|![maxbin_acc](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/430a8544-1e90-40d7-9285-3ce4f3155b86)


Wavelet-based algorithm for similarity with real falls data that you can find in `feature_extraction\file_csv.zip` file.

| <!-- -->    | <!-- -->    | <!-- -->    | 
|-------------|-------------|-------------|
![peaks4](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/be1b1607-daed-4e9a-8fb3-4e23d47d2bd0)|![peaks5](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/9f3e9a69-846a-41a7-8c98-dcdc5f91c85a)|![peaks6](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/b03283e2-0238-4b1f-9ed8-e4fd4b871dfa)
![peaks1](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/7adb06b1-f752-4307-b484-0a36f97c7230)|![peaks2](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/ddbb92e2-5725-4e84-8332-df6c682df9b2)|![peaks3](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/cb8bfe3a-b962-4c54-a065-e504be80d91e)

The final data set will consist of the following features:
>- 5 peaks of the magnitude of the power spectrum density for the accelerometer signal
and 5 peaks for the gyroscope;
>- 3 statistics (median, mad and skewness) for the two PSDs;
>- 1 cwt coefficient for the similarity with the accelerometer peak mother wavelet.
8

## Binary classification

For simplicity we started with a simply classification of signals with or without falls using several ML algorithms.

Here a brief summary display of our results:

![dec_boundaries](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/7f31bcd9-5a7c-45d7-8c82-2c0dca949475)

| <!-- -->    | <!-- -->    |
|-------------|-------------|
![roc_curves](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/c4023113-4454-4194-8518-0a2ef91e6bc1)|![calibration_plot](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/02c3f652-985a-45dc-b42e-249caa3c8776)

## Multi-class classification

After scaling, resampling with a K-Means SMOTE and selecting a coherent number of classes through an Unsupervised learning approach we got the following results:

1) The best number of classes according to a K-Means-based cluster analysis is 6 and, according to their coherence score we collapsed `fall` and `light` to the same label.

![hist_sl](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/5ae43dba-d742-4534-827d-e9f8800de056)

2) After the above analysis we fitted several ML algorithms:

| KNN   | SVM    | Softmax | 
|-------------|-------------|-------------|
![knn_final](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/a178d10b-c9b2-4c42-967a-4f4e6e8de4ab)|![svm_final](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/c8e6f952-deee-4951-8770-1e21b422e71d)|![softmax_final](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/bb97a4b8-2eae-4b01-a90a-cce857088d8b)

| AdaBoost    | Random forest    | Perceptron    | 
|-------------|-------------|-------------|
![adaboost_final](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/0648cd06-0a77-4f59-96fd-a91e8b494f3f)|![rf_final](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/420538d0-ff9b-4d1d-bc28-151f40c21477)|![percptron_final](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/ab6b6587-d132-4605-a033-5e677e4b8ddf)

## Multi-class classification with deep learning

We therefore decided to try a series of neural architectures that did not require the use of special feature extraction techniques. As we expected, we obtained results comparable or superior to the best classical machine learning models.

We noticed how, therefore, the deep learning approach avoids great efforts to extract fundamental features from the signals, at the expense of a lack of interpretability of the results.

Here we report the results for our best deep learning model: the **CBAM-EDU (Convolutional-Bidirectional LSTM with Educational Enhancements)**.

![NN_res](https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/f245fb5d-9efc-4aee-b31f-96a39391c175)
