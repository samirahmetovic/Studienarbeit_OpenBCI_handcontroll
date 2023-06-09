<img src="https://abload.de/img/eeg_c500vjes9.png" width="200px">

# OpenBCI Handcontrol

This project aims to use OpenBCI to create a hand control system that can be used to control devices or applications by detecting hand gestures from EEG signals. The project consists of three main components: data collection, data preprocessing, and classification.

# Data Collection

Data collection is performed using OpenBCI Cyton board and EEG electrodes. In order to train the classification model, the user will open and close their hand while the EEG signals are recorded. The collected data will then be used for model training and testing.

The scripts for collecting Data is create_training_data4.py

# Data Preprocessing

Before feeding the data into the classification model, some preprocessing steps are performed on the raw EEG signals. These steps include denoising, transformation, and filtering. The denoising step removes unwanted noise from the signals, while the transformation step converts the signals into a format suitable for the CNN model. Finally, filtering is used to remove any unwanted frequencies from the signals.

There is the mr_clean.py which does a wavelet transformation
and the mr_clean-fft.py which does a FFT instead

# Classification

The classification model used in this project is a convolutional neural network (CNN) implemented in Pytorch. The trained model is then used to classify the EEG signals into hand gestures, such as open and close hand. The model is trained using the preprocessed data, and the accuracy of the model is evaluated using test data.

In connection with the "Studienarbeit" there are following models:

nn_model.py was the model of the first try. it was an basic ANN with fully connected Layers.

our second try was the CNN from nn_model_7.py

the third try was the EEGNET in nn_model_9.py for wavelet data
and nn_model_9_1.py for FFT data

keep in mind, that the nn*model*... have their own scripts for training
nn_model.py -> train_model.py -> test_model.py
nn_model_7.py -> train_model_7.py -> test_model_7.py
...
