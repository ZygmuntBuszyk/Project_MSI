import numpy as np

# Custom classifier
from Custom.Knn import customKnn

# Others
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, RepeatedKFold
# Suppress warnings
import warnings  #Warn the user when attempting for a wrong number for 'K'
warnings.simplefilter("ignore")

#Database
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

# Standard scaler - skalowanie cech
from sklearn.preprocessing import StandardScaler
# PCA - zmniejszenie liczby wymiarów
from sklearn.decomposition import PCA

# GridSearch
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

# cechy 
X = data.data
# etykiety  0,1
y = data.target


# skalowanie cech 
scaler = StandardScaler()
scaler.fit(X) # StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.mean_
X = scaler.transform(X)

# zmniejszenie wymiarów zbioru cech
pca = PCA(n_components = 'mle')
pca.fit(X)

X = pca.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3 ,train_size=0.7, random_state=0)

tuned_params = {"k" : [3,5,7,9,11]}
gs = GridSearchCV(customKnn(), tuned_params)
gs.fit(X_test,y_test)
gs.best_params_ 
print(gs.best_params_)



knn = customKnn(k=3)
score =[]
nsp = []
RKF = RepeatedKFold(n_splits=2, n_repeats=5)

for train_index, test_index in RKF.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test =  y[train_index], y[test_index]
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))
    nsp.append((y_test != y_pred).sum()/len(y_test))

accuracy = np.mean(score) 
standardDeviation = np.std(score) 
mean_error = np.mean(nsp)
print(score)


print('---------------------------------------------------------')
print ("Srednia dokładnosc klasyfikacji: {0}".format(accuracy))
print ("Odchylenie standardowe wyników: {0}".format(standardDeviation))
print ("Uśredniony błąd nieprawidłowej klasyfikacji : {0} %".format(mean_error*100))
print('---------------------------------------------------------')
