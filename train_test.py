#!/usr/bin/env/python

import numpy as np
import random

import classifiers
from pca import pca

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# features vector
path = ".\\Data\\"
features = np.load(path + "features_vector.npy")

# Apply or not PCA on features
PCA = False
if PCA:
    features, N = pca(features)
    feat = N # number of features
else:
    feat = 66 # Number of features for MFCC's + ZCR + AP

# Genre - division features
genres = {  'Blues' : features[0:100 , :], 'Classical' : features[100:200 , :],
            'Country' : features[200:300 , :], 'Disco' : features[300:400 , :],
            'Hiphop' : features[400:500 , :], 'Jazz' : features[500:600 , :],
            'Metal' : features[600:700 , :], 'Pop' : features[700:800 , :],
            'Reggae' : features[800:900 , :], 'Rock' : features[900:1000 , :]
         }
classes = { 'Blues' : 0,  'Classical' : 1, 'Country' : 2, 'Disco' : 3, 'Hiphop' : 4, 'Jazz' : 5, 'Metal' : 6, 'Pop' : 7, 'Reggae' : 8, 'Rock' : 9 }

# Scatter plot of the first 3 features
fig = plt.figure()
ax = Axes3D(fig)

for k in genres.keys():
    ax.scatter(genres[k][:,0], genres[k][:,1], genres[k][:,2], label = k)

fig.legend()
plt.show()

# Labels generation. Training Set : 75 samples/class ; Testing Set: 25 samples/class
y_train = np.empty((1,0))
y_test = np.empty((1,0))

for g in genres.keys():
    y_train = np.append(y_train, [g] * 75)
    y_test = np.append(y_test, [g] * 25)

#####################################################################################

# Test on 100 classifications, get mean result
n_classifications = 100

conf_matrix = np.zeros((10,10))

# Classify for n_classifications times 
for c in range(n_classifications):

    print("Classification Number:", (c+1))

    # Generating training and testing set
    x_train = np.empty((0,feat))
    x_test = np.empty((0,feat))

    for genere in genres.keys():
        np.random.shuffle(genres[genere])
        x_train = np.append(x_train, (genres[genere])[0:75, :] , axis = 0)
        x_test = np.append(x_test, (genres[genere]) [75:100, :] , axis = 0)

    # predictions for each classifier

    # SVM improves with KNN
    svm_predictions = classifiers.svm(x_train, y_train, x_test) 

    # Adaptive Boosting
    ada_predictions = classifiers.boost(x_train, y_train, x_test)

    # Naive Baesian
    bay_predictions = classifiers.bayes(x_train, y_train, x_test)

    # Choose final prediction with rules:
    # 1. If baesian states one between "Classical", "Jazz", "Metal", "Reggae" pick it
    # 2. else if every classifier or 2/3 state the same pick that choice 
    # 3. If every classifier states something different focus on Boost and SVM:
    #           if both state on of: "Disco", "Jazz", "Metal", "Pop" or "Reggae" pick SVM
    #           else if both state one of others pick Boost
    #           else pick random between the two
    final_prediction = []
    for i in range ( len(svm_predictions) ):

        if (bay_predictions[i] == 1) or (bay_predictions[i] == 5) or (bay_predictions[i] == 6) or (bay_predictions[i] == 8):
            final_prediction.append(bay_predictions[i])

        elif (bay_predictions[i] == svm_predictions[i]) or (bay_predictions[i] == ada_predictions[i]):
            final_prediction.append(bay_predictions[i])
            
        elif (ada_predictions[i] == svm_predictions[i]):
            final_prediction.append(ada_predictions[i])

        elif (  (ada_predictions[i] == 3) or (ada_predictions[i] == 5) or (ada_predictions[i] == 6) or (ada_predictions[i] == 7) or 
            (ada_predictions[i] == 8) ) and ( (svm_predictions[i] == 3) or (svm_predictions[i] == 5) or (svm_predictions[i] == 6) or 
            (svm_predictions[i] == 7) or (svm_predictions[i] == 8) ):
            final_prediction.append(svm_predictions[i])

        elif (  (ada_predictions[i] == 0) or (ada_predictions[i] == 1) or (ada_predictions[i] == 2) or (ada_predictions[i] == 4) or 
            (ada_predictions[i] == 9) ) and ( (svm_predictions[i] == 0) or (svm_predictions[i] == 1) or (svm_predictions[i] == 2) or 
            (svm_predictions[i] == 4) or (svm_predictions[i] == 9) ):
            final_prediction.append(ada_predictions[i])
        
        else:
            final_prediction.append( random.choice ( [ada_predictions[i], svm_predictions[i]] ) )

    for i in range( len(final_prediction)):
        conf_matrix[ classes[y_test[i]], final_prediction[i] ] += 1

# Results
conf_matrix[:,:] /= (250 * n_classifications) 
conf_matrix[:,:] *= 100 

accuracy = np.sum(conf_matrix.diagonal())/np.sum(conf_matrix)

precision = []
recall = []
for i in range(10):
  precision.append(conf_matrix[i,i]/ np.sum(conf_matrix[:,i]))
  recall.append(conf_matrix[i,i]/ np.sum(conf_matrix[i,:]))

precision = np.mean(np.asarray(precision))
recall = np.mean(np.asarray(recall))

plt.imshow(conf_matrix, vmin = 0, vmax = 10, cmap='YlGnBu')
plt.colorbar()
plt.xlabel("Predicted Values")
plt.xticks([0,1,2,3,4,5,6,7,8,9],['Blues',  'Classical', 'Country', 'Disco', 'HipHop', 'Jazz','Metal', 'Pop', 'Reggae', 'Rock'])
plt.yticks([0,1,2,3,4,5,6,7,8,9], ['Blues',  'Classical', 'Country', 'Disco', 'HipHop', 'Jazz','Metal', 'Pop', 'Reggae', 'Rock'])
plt.ylabel("Real Values")
plt.title("Accuracy: " + "{0:.2f}".format(accuracy*100) + "%, Precision: " + "{0:.2f}".format(precision*100) + "%, Recall: " + "{0:.2f}".format(recall*100) + "%" )

for (i, j), z in np.ndenumerate(conf_matrix):
    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

plt.show()