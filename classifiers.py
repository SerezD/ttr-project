import numpy as np
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.ensemble import AdaBoostClassifier

n_class = 10
classes = { 0 : 'Blues',  1 : 'Classical', 2 : 'Country', 3 : 'Disco', 4 : 'Hiphop', 5 : 'Jazz', 6 : 'Metal', 7 : 'Pop', 8 : 'Reggae', 9 : 'Rock'}

# SVM improved with KNN
def svm(x_train, y_train, x_test) :

    # Initializing models
    models = [SVC(kernel='linear', probability=True) for _ in range(n_class) ]

    # Learning ... Binary classifiers
    for i in range(n_class):
        models[i].fit(x_train, y_train== classes[i])

    # Initial predictions (predictions for each classifier)
    prediction_scores = []
    for i in range(n_class):
        prediction_scores.append(models[i].predict_proba(x_test)[:,1])

    prediction_scores = np.asarray(prediction_scores)
  
    # KNN improvement on similar probabilities
  
    epsilon = 0.05
    predictions = []

    # Confirm or change prediction for each test point
    for i in range( np.shape(x_test)[0] ):

        # Predictions on this point and best_prediction
        point_predictions = prediction_scores[:,i]
        best = max(point_predictions)

        best_points = [ p for p in point_predictions if p >= best - epsilon ]

        # Predition confirmed
        if len( best_points ) == 1:
            predictions.append(np.where(prediction_scores[:,i] == best_points[0])[0][0])
    
        # KNN on best prediction classes
        else:

            # Find the classes with the best predictions
            # Note: it's possible to have the same prediction for two classes
            best_classes = []
            index = 0
            for p in best_points:
                best_classes.append(np.where(point_predictions == p)[0][0] + index)
                index += 1
                point_predictions = np.delete(point_predictions, np.where(point_predictions == p)[0][0])

            # get the train points in those best classes
            best_train = np.empty((0, np.shape(x_train)[1] ))
            points = (int) (np.shape(x_train)[0] / n_class) # Supposing al classes have same number of points

            for j in best_classes:
                best_train = np.append ( best_train,  x_train[ points * j : (points * j) + points , : ], axis = 0) 

            # point to test
            sample = x_test[i, :] 
            
            # Euclidean distance
            euc = cdist(best_train, [sample], metric='euclidean')
            neighbours = np.argsort(euc, axis = 0)
            k_neigh = (neighbours[:3,:])[: , 0] # 3NN 

            labels = []
            for n in k_neigh:
                labels.append(best_classes[(int)(n / points)])
 
            predictions.append(stats.mode(labels , axis=0)[0][0])

    return predictions

# Adaptive Boosting with default base classifier
def boost(x_train, y_train, x_test) :

    prediction_scores = []   

    # Binary classifier
    for i in range(n_class):
      booster = AdaBoostClassifier(n_estimators=500)
      booster.fit(x_train, y_train == classes[i])
      prediction_scores.append(booster.predict_proba(x_test)[:,1])
    
    prediction_scores = np.asarray(prediction_scores)

    return np.argmax(prediction_scores,axis=0) 

# Naive Baesian
def bayes(x_train, y_train, x_test) :

    n_points = (int) (np.shape(x_train)[0] / n_class) # Supposing al classes have same number of points
    probabilities = np.empty((0,np.shape(x_test)[0]))

    for c in range(n_class):

        # Get the class
        train_class = x_train[c*n_points : (c*n_points) + n_points, :]

        # Mean and Covariance
        mu = np.mean(train_class, axis=0)
        cov = np.cov(train_class.T)

        # Probability that every sample belongs to this class. Prior = 1 / (n_class)
        probabilities = np.append(probabilities,  [stats.multivariate_normal.pdf(x_test,mu,cov, allow_singular=True) * (1/n_class)], axis = 0)

    return np.argmax(probabilities, axis = 0) 