# ttr-project

# Data
Data contains the feature vector generated from feature_extraction.py and can be used by train_test.py

# feature_extraction
This code extracts from a dataset the features vector and saves it in .npy format.
The dataset used is [GTZAN](https://www.kaggle.com/carlthome/gtzan-genre-collection) where every genre is one directory.
The default features extracted are 20 MFCC's mean, variance and skew; ZCR Mean and Variance and peaks information from Autocorrelation Plot (total 66 features).

# train_test
Executing this generates the confusion matrix of the classifiers; trained and tested on features from feature_extraction.py.
Classifiers are taken from the classifiers.py code, and you can also execute PCA on features, setting the flag to True (in train_test code).
Also, the visualization of first three dimensions of features is visualized.
