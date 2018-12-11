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
    regr_1.fit(X_train, y_train)

    # Predict
    y_pred = regr_1.predict(X_test)
    f = open("./results/regression_tree.txt", "w")
    f.write(f"MSE : {mean_squared_error(y_test, y_pred)}\nR^2 score : {r2_score(y_test, y_pred)}")
    try:
        generate_decision_tree(regr_1, "regression_tree")
    except:
        print('Error generating decision tree')


def get_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = tree.DecisionTreeRegressor(max_depth=4)
    tree_model = model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    model.decision_path(X_train)
    f = open("./results/decision_tree.txt", "w")
    f.write(f"MSE : {mean_squared_error(y_test, y_pred)}\n" +
            f"R^2 score : {r2_score(y_test, y_pred)}")
    try:
        # You should install graphviz on your system to generate decision tree otherwise you will get exception
        generate_decision_tree(model, "decision_tree")

    except:
        print('Error generating decision tree')


def generate_decision_tree(model, name):
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=get_features(),
                                    class_names="class",
                                    filled=True, rounded=True,

                                    special_characters=True)
    import graphviz
    graph = graphviz.Source(dot_data)
    graph.render(filename=name)


def get_bagging(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    bagging = BaggingClassifier()
    bagging_model = bagging.fit(X_train, y_train)

    y_pred = bagging_model.predict(X_test)
    print("MSE error is :", mean_squared_error(y_test, y_pred))
    print("r2 score is :", r2_score(y_test, y_pred))

    f = open("./results/bagging.txt", "w")
    f.write(f"Confusion matrix : {confusion_matrix(y_pred, y_test)}\nMSE : {mean_squared_error(y_test, y_pred)}\n" +
            f"Accuracy : {accuracy_score(y_test, y_pred)}\nR^2 score : {r2_score(y_test, y_pred)}")


def get_random_forrest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    random_forrest = RandomForestClassifier()
    model = random_forrest.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("MSE error is :", mean_squared_error(y_test, y_pred))
    print("r2 score is :", r2_score(y_test, y_pred))

    f = open("./results/random_forrest.txt", "w")
    f.write(f"Confusion matrix : {confusion_matrix(y_pred, y_test)}\nMSE : {mean_squared_error(y_test, y_pred)}\n" +
            f"Accuracy : {accuracy_score(y_test, y_pred)}\nR^2 score : {r2_score(y_test, y_pred)}")


def get_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    svmachine = svm.SVC(gamma='auto', kernel='linear')
    svm_model = svmachine.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    f = open("./results/svm.txt", "w")
    f.write(f"Confusion matrix : {confusion_matrix(y_pred, y_test)}\n"
            f"MSE : {mean_squared_error(y_test, y_pred)}\nAccuracy : {accuracy_score(y_test, y_pred)}"
            f"\nR^2 score : {r2_score(y_test, y_pred)}")


X = df[get_features()]
y = df["class"]

get_decision_tree(X,y)
get_scores_for_regression_tree(X,y)
get_svm(X, y)
get_bagging(X, y)
get_random_forrest(X, y)
