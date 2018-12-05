import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from markdown import markdown
import warnings
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("./breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def get_features():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    return features





def get_scores_for_regression_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X_train, y_train)
    regr_2.fit(X_train, y_train)

    # Predict
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    print("MSE error is :", mean_squared_error(y_test, y_1))
    print("r2 score is :", r2_score(y_test, y_1))
    generate_decision_tree(regr_1, "regression_tree")



def get_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = tree.DecisionTreeRegressor(max_depth=5)
    tree_model = model.fit(X_train, y_train)
    predict = tree_model.predict(X_test)
    model.decision_path(X_train)
    print("MSE error is :", mean_squared_error(y_test, predict))
    print("r2 score is :", r2_score(y_test, predict))

    path1 = model.decision_path(X_test)

    from sklearn.externals.six import StringIO
    from sklearn.tree import export_graphviz
    import pydotplus

    # dot_data = StringIO()

    # graph.render(view=True)
    # graph.save("tree.jpg")
    generate_decision_tree(model, "decision_tree")

def generate_decision_tree (model, name):
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=get_features(),
                                    class_names="class",
                                    filled=True, rounded=True,

                                    special_characters=True)
    import graphviz
    graph = graphviz.Source(dot_data)
    graph.render(filename=name)


def get_bagging(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    bagging = BaggingClassifier()
    bagging_model = bagging.fit(X_train, y_train)

    predict = bagging_model.predict(X_test)
    confusion = confusion_matrix(y_test, predict)
    print(confusion)
    print("MSE error is :", mean_squared_error(y_test, predict))
    print("r2 score is :", r2_score(y_test, predict))

def get_svm(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    svmachine = svm.SVC(gamma='auto', kernel='linear')
    svm_model = svmachine.fit(X_train, y_train)

    print(svm_model.support_)
    print(svm_model.support_vectors_)
    print(svm_model.n_support_)
    # print(svm_model.coef_)

    # print(svm_model.predict_proba(X_test))
    # print(svm_model.predict_log_proba(X_test))
    print(svm_model.score(X_test, y_test))
    y_pred = svm_model.predict(X_test)

    ########Confusion matrix
    cnf_matrix = confusion_matrix(y_pred, y_test)
    print(cnf_matrix)

    #####Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # print("Precision:", precision_score(y_test, y_pred))
    # print("Recall:", recall_score(y_test, y_pred))
    # print("F_measure:", f1_score(y_test, y_pred))



X = df[get_features()]
y = df["class"]


# print(get_decision_tree(X,y))
# print(get_decision_tree(X,y))
get_decision_tree(X,y)
get_scores_for_regression_tree(X,y)
# get_svm(X,y)