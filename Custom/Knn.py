import numpy as np

from math import sqrt
from collections import Counter

import warnings #Warn the user when attempting for a wrong number for 'K'
from sklearn.base import BaseEstimator, ClassifierMixin
class customKnn(BaseEstimator, ClassifierMixin) : 

    def __init__(self, k=5, prediction=None, fitted=False): # constructor
        self.k = k
        self.prediction = prediction
        self.fitted = fitted

    def fit(self, X_train, y_train) :
        if self.k > len(X_train):
		    # raise ValueError
            print("K/n_neighbors ma wartosc wieksza od grupy glosujacych")
            return

        self.X_train = X_train
        self.y_train = y_train
        self.fitted = True
        return self

    def predict(self, X_test) :
        if self.fitted :
            # create list for distances and targets
            predictions = []
             # predict for each testing observation
            for i in range(len(X_test)):
                x_test = X_test[i, :]
                distances = []
                targets = []

                for i in range(len(self.X_train)):
                    
                # first we compute the euclidean distance
                    euclidean_distance = np.sqrt(np.sum(np.square(x_test - self.X_train[i, :])))
                    # add it to list of distances
                    distances.append([euclidean_distance, i])

                # sort the list
                distances = sorted(distances)

                # make a list of the k neighbors' targets
                vote = []

                for i in range(self.k):
                    index = distances[i][1]
                    targets.append(self.y_train[index])
                    vote.append(self.y_train[index])
                
                # return most common target
                most_common_target = Counter(targets).most_common(1)[0][0]
                confidence = Counter(targets).most_common(1)[0][1] / self.k

                if confidence <= 0.6:
                    new_votes = []
                    new_k = self.k * 2 + 1
                    for i in range(new_k):
                        index = distances[i][1]
                        # targets = votes
                        new_votes.append(self.y_train[index])
                    new_most_common = Counter(new_votes).most_common(1)[0][0]
                    new_confidence = Counter(new_votes).most_common(1)[0][1] / new_k  
                 
                    most_common_target =  new_most_common

              
                predictions.append(most_common_target)
            return np.asarray(predictions)

        else :
            print("This KNeighborsClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")