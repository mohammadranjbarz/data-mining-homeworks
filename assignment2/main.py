import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix

df = pd.read_csv("./breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def get_features():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    return features


def get_model_report(y, y_pred):
    mis_classification = calculate_mis_classification(y, y_pred)
    return "MisClassification  = " + str(mis_classification) + "\n\n" + "Accuracy  = " + str(
        1 - mis_classification) + "\n\n" + "F1 score  =  " + str(
        f1_score(y, y_pred, average="macro")) + "\n\n" + "Precision score  =  " + str(
        precision_score(y, y_pred, average="macro")) + "\n\n" + "Recall score  =  " + str(
        recall_score(y, y_pred, average="macro")) + "\n\n"


def save_logistic_regression(X, y):
    logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    f = open("./results/LogisticRegression.txt", "w")
    f.write(get_model_report(y, y_pred))


def save_qda(X, y):
    qda = QDA().fit(X, y)
    y_pred = qda.predict(X)
    f = open("./results/Qda.txt", "w")
    f.write(get_model_report(y, y_pred))


def save_lda(X, y):
    lda = LDA().fit(X, y)
    y_pred = lda.predict(X)
    f = open("./results/Lda.txt", "w")
    f.write(get_model_report(y, y_pred))


def save_gnb(X, y):
    gnb = GaussianNB().fit(X, y)
    y_pred = gnb.predict(X)
    f = open("./results/gnb.txt", "w")
    f.write(get_model_report(y, y_pred))


def calculate_mis_classification(y, y_pred):
    y = y.values
    misclassification_sum = 0
    for i in range(len(y)):
        misclassification_sum += 1 if y[i] != y_pred[i] else 0
    misclassificationError = misclassification_sum / len(y_pred)
    return round(misclassificationError, 4)


X = df[get_features()]
y = df["class"]
#### Phase1 of Assignment2
save_logistic_regression(X, y)
save_qda(X, y)
save_lda(X, y)
save_gnb(X, y)


#### Phase2 of Assignment3
def save_linear_regression(X, y):
    linear_regression = LinearRegression().fit(X, y)
    y_pred = linear_regression.predict(X)
    y_pred_classified = []
    for i in range(len(y_pred)):
        if (y_pred[i] > 3):
            y_pred_classified.append(4)
        else:
            y_pred_classified.append(2)
    f = open("./results/LinearRegression.txt", "w")
    f.write(get_model_report(y, y_pred_classified))


save_linear_regression(X, y)
