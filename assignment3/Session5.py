############## Classification#############

import pandas as pd
import numpy as np


###reading data
# missValue=['?']
Autism=pd.read_csv('Autism-Adult.csv')
# print(Autism.isnull().sum())
# print(Autism['age numeric'][62])
nanind=[]
for ind in range( len(Autism['age numeric'])):
		if (Autism['age numeric'][ind]=='?'):
			nanind.append(ind)

Autism=Autism.drop(nanind)
X=Autism.values[:,:12]
y=Autism.values[:,12]
y_edit= np.array([1 if yinstance=='yes' else 0 for yinstance in y ])
X[:,11]=np.array([1 if xinstance=='f' else 0 for xinstance in X[:,11] ])



###### Devide data to test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y_edit,test_size=0.2,random_state=0)


###########Logistic Regression#####
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logregmodel = logreg.fit(X_train, y_train)
y_pred = logregmodel.predict(X_test)



########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred , y_test )
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))





##########Naive Bayes #########
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()
GNBmodel = GNB.fit(X_train, y_train)
y_pred = GNBmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred,y_test )
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F_measure:",metrics.f1_score(y_test, y_pred))



######QDA ###########
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
QDA= QuadraticDiscriminantAnalysis()
QDAmodel = QDA.fit(X_train, y_train)
y_pred = QDAmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




######LDA ###########
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA= LinearDiscriminantAnalysis()
LDAmodel = LDA.fit(X_train, y_train)
y_pred = LDAmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred ,y_test )
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))



###### Classification with Linear Regression######
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LRmodel = LR.fit(X_train, y_train)
y_regpred = LRmodel.predict(X_test)
y_pred= [1 if x>=0.4 else 0 for x in y_regpred]

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred , y_test)
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
   # print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   # print(X_train, X_test, y_train, y_test)





1+1