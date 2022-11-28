from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB 

from sklearn.preprocessing import minmax_scale 
from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import matthews_corrcoef 
from sklearn.metrics import f1_score 
from sklearn.metrics import plot_roc_curve

col_names =["Artist Name","Track Name","Popularity","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_in min/ms","time_signature","Class"]
dataset = pd.read_csv('train.csv',col_names)   


dataset = dataset.dropna() 
dataset = dataset.drop(['Artist Name'], axis=1)
dataset = dataset.drop(['Track Name'], axis=1) 

x = dataset.drop(['Class'], axis = 1).values 

y = dataset['Class'].values 

scaler = StandardScaler() 
scaler.fit(x) 
x= scaler.fit_transform(x) 


pca = PCA(n_components=14, svd_solver='full')
pca.fit(x) 
#print(pca.explained_variance_ratio_)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)

x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test) 

logistic_regression = LogisticRegression(solver='newton-cg', max_iter=1000, multi_class='multinomial')  

logistic_regression.fit(x_train,y_train) 
y_pred_lr= logistic_regression.predict(x_test) 
y_predproba_lr= logistic_regression.predict_proba(x_test) 
mcc_lr = metrics.matthews_corrcoef(y_test, y_pred_lr)
#roc_auc_lr = metrics.roc_auc_score(y_test, y_pred_lr, multi_class='ovr') #Roc auc 
f1_lr = f1_score(y_test, y_pred_lr, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn') 
"""roc = {label: [] for label in multi_class_series.unique()} 
for label in multi_class_series.unique(): 
    logistic_regression.fit(x_train, y_train == label) 
    predictions_proba = logistic_regression.predict_proba(x_test) 
    roc[label] += roc_auc_score(y_test, predictions_proba[:,1])"""
print("MCC Logistic Regression:", mcc_lr) 
#print("Roc_auc score:", roc_auc_lr) 
print("F1 score Logistic Regression:", f1_lr)


c_svc=10
kernel= 'rbf'
svc = SVC(C=c_svc, kernel=kernel, gamma=0.01) 
svc.fit(x_train, y_train) 
y_pred_svc = svc.predict(x_test)
mcc_svc = metrics.matthews_corrcoef(y_test, y_pred_svc) 
#roc_auc_knn = metrics.roc_auc_score(y_test, y_pred_knn, multi_class='ovr') #Roc auc
f1_svc = f1_score(y_test, y_pred_svc, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn') #F1 score
print("MCC Logistic Regression:", mcc_svc)
#print("Roc_auc score:", roc_auc_knn)
print("F1 score Logistic Regression:", f1_svc)


n_neighbors = 75 
knn = KNeighborsClassifier(n_neighbors,weights='uniform' ,metric='euclidean', metric_params=None,algorithm='brute') 
knn.fit(x_train, y_train) 

y_pred_knn = knn.predict(x_test)  

"""cfsion_mtx_knn = metrics.confusion_matrix(y_test, y_pred_knn) #Matriz de confusion
TP_knn = cfsion_mtx_knn[1, 1]
TN_knn = cfsion_mtx_knn[0, 0]
FP_knn = cfsion_mtx_knn[0, 1]
FN_knn = cfsion_mtx_knn[1, 0]"""

mcc_knn = metrics.matthews_corrcoef(y_test, y_pred_knn) 
#roc_auc_knn = metrics.roc_auc_score(y_test, y_pred_knn, multi_class='ovr') #Roc auc
f1_knn = f1_score(y_test, y_pred_knn, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn') 


print("MCC Logistic Regression:", mcc_knn)
#print("Roc_auc score:", roc_auc_knn)
print("F1 score Logistic Regression:", f1_knn) 