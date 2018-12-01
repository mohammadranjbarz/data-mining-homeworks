from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import metrics

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


##############  SVM #########
clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
 coef0=0.0, shrinking=True, probability=False, tol=0.001,
 cache_size=200, class_weight=None, verbose=False, max_iter=-1,
 decision_function_shape='ovr', random_state=None)
clf.fit(X_train, y_train)

#C: cost or penalty parameter

#kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

#degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

#gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

#shrinking: Whether to use the shrinking heuristic.

#probability: Whether to enable probability estimates.
#  This must be enabled prior to calling fit, and will slow down that method.
#decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’


svmachine = svm.SVC(gamma='auto',kernel='linear')
svm_model = svmachine.fit(X_train, y_train)

print(svm_model.support_)
print(svm_model.support_vectors_)
print(svm_model.n_support_)
# print(svm_model.coef_)

# print(svm_model.predict_proba(X_test))
# print(svm_model.predict_log_proba(X_test))
print(svm_model.score(X_test,y_test))
y_pred = svm_model.predict(X_test)

########Confusion matrix
cnf_matrix = metrics.confusion_matrix(y_pred,y_test )
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F_measure:",metrics.f1_score(y_test, y_pred))
print( "classification error is :", np.sum(svm_model.predict(X_test) != y_test) / len(y_test) )


###Custom kernel
def my_kernel(X, Y):
    return np.dot(X, Y.T)

svm_custom_model = svm.SVC(kernel=my_kernel)




######## Decision Tree ########
from sklearn import tree

enb_frame = pd.read_csv('ENB2012_data.csv')
enb=enb_frame.values
X = enb[:768,:8]
y = enb[:768,8]

X_train, X_test, y_train,y_test= train_test_split(X, y,test_size=0.2 )

model = tree.DecisionTreeRegressor(max_depth=5)
tree_model = model.fit(X_train, y_train)
predict = tree_model.predict(X_test)

### print(enb_frame)
# import graphviz
# dot_data = tree.export_graphviz(tree_model, out_file="ghgh.txt")
# graph = graphviz.Source(dot_data)
# graph.render("ENB2012",view=True)

# metrics
from sklearn.metrics import r2_score, mean_squared_error
print("MSE error is :", mean_squared_error(y_test, predict))
print("r2 score is :", r2_score(y_test, predict))

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
X_train,X_test,y_train,y_test=train_test_split(X,y_edit,test_size=0.2,random_state=0)


### tree based methods for classification
# criterion = 'gini' or 'entropy'
tree_classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
tree_model = tree_classifier.fit(X_train, y_train)
predict = tree_model.predict(X_test)
#
# import graphviz
# dot_data = tree.export_graphviz(tree_model, out_file="ghgh.txt")
# graph = graphviz.Source(dot_data)
# graph.render("Autism",view=True)
#
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
# dot_data = StringIO()
# export_graphviz(tree_model , out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

####Confusion matrix
cnf_matrix = metrics.confusion_matrix(predict,y_test )
print(cnf_matrix)

###Metrics
print("Accuracy:",metrics.accuracy_score(y_test, predict))
print("Precision:",metrics.precision_score(y_test, predict))
print("Recall:",metrics.recall_score(y_test, predict))
print("F_measure:",metrics.f1_score(y_test, predict))
print( "classification error is :", np.sum(predict != y_test) / len(y_test) )




########## Bagging method  ######
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
bagging = BaggingClassifier()
bagging_model = bagging.fit(X_train, y_train)

bagging_predict = bagging_model.predict(X_test)
confusion = metrics.confusion_matrix(y_test, bagging_predict)

from colorama import Back, Fore
print(Back.YELLOW)
print("\t\t{0:s} \t {1:s} ".format(Fore.RED + "predicted NO", Fore.RED + "predicted YES"))
print(Fore.GREEN + "actual NO \t {0} \t\t  {1} \t".format(Fore.BLUE + str(confusion[0, 0]), Fore.BLACK + str(confusion[0, 1])))
print(Fore.GREEN + "actual YES \t {0} \t\t  {1} \t".format(Fore.BLACK + str(confusion[1, 0]), Fore.BLUE + str(confusion[1, 1])))

print(Back.CYAN)
print(metrics.classification_report(y_test, bagging_predict))
print(Back.RESET)



########### Random Forest  ###########

random_forest = RandomForestClassifier(min_samples_split=5, min_samples_leaf=2, max_depth=10)
random_forest_model = random_forest.fit(X_train, y_train)

random_forest_predict = random_forest_model.predict(X_test)
print( metrics.confusion_matrix(y_test, random_forest_predict))

print(Back.GREEN)
print(metrics.classification_report(y_test, random_forest_predict))
print(Back.RESET)


1+1