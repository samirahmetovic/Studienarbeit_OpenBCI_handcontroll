![EEG](https://abload.de/img/eeg_c500vjes9.png)

# OpenBCI Handcontrol

This project aims to use OpenBCI to create a hand control system that can be used to control devices or applications by detecting hand gestures from EEG signals. The project consists of three main components: data collection, data preprocessing, and classification.

# Data Collection

Data collection is performed using OpenBCI Cyton board and EEG electrodes. In order to train the classification model, the user will open and close their hand while the EEG signals are recorded. The collected data will then be used for model training and testing.

# Data Preprocessing

Before feeding the data into the classification model, some preprocessing steps are performed on the raw EEG signals. These steps include denoising, transformation, and filtering. The denoising step removes unwanted noise from the signals, while the transformation step converts the signals into a format suitable for the CNN model. Finally, filtering is used to remove any unwanted frequencies from the signals.

# Classification

The classification model used in this project is a convolutional neural network (CNN) implemented using Pytorch. The trained model is then used to classify the EEG signals into hand gestures, such as open and close hand. The model is trained using the preprocessed data, and the accuracy of the model is evaluated using test data.
