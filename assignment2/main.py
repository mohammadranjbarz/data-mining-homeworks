import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("./breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')

def getFeatures():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    return features

# def get_formatted_data_frame_from_predictions(X, y, predictions, params, features):
#     newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
#     MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))
#     var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
#     sd_b = np.sqrt(var_b)
#     ts_b = params / sd_b
#     p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
#     sd_b = np.round(sd_b, 3)
#     ts_b = np.round(ts_b, 3)
#     p_values = np.round(p_values, 3)
#     params = np.round(params, 4)
#     myDF3 = pd.DataFrame()
#     features.insert(0, "constants")
#     myDF3["Feature"], myDF3["Coefficients"], myDF3["t values"], myDF3["Standard Errors"], myDF3["Probabilites"] = [
#         features, params, ts_b, sd_b, p_values]
#     return MSE, myDF3

def saveLogisticRegression():
    X = df[getFeatures()]
    y = df["class"]
    logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    y_pred = logreg.predict(X)
    logesticRscore = logreg.score(X, y)
    f = open("./results/LogisticRegression.txt", "w")
    f.write("Misclassification  = " + str(calculateMisclassification(y, y_pred)) + "\n\n")
    f.write("R-squared  = " + str(logesticRscore) + "\n\n")


def saveQda():
    X = df[getFeatures()]
    y = df["class"]
    qda = QDA().fit(X, y)
    y_pred = qda.predict(X)
    qdaRscore = qda.score(X, y)
    f = open("./results/Qda.txt", "w")
    f.write("Misclassification  = " + str(calculateMisclassification(y, y_pred)) + "\n\n")
    f.write("R-squared  = " + str(qdaRscore) + "\n\n")


def saveLda():
    X = df[getFeatures()]
    y = df["class"]
    lda = LDA().fit(X, y)
    y_pred = lda.predict(X)
    ldaRscore = lda.score(X, y)
    f = open("./results/Lda.txt", "w")
    f.write("Misclassification  = " + str(calculateMisclassification(y, y_pred)) + "\n\n")
    f.write("R-squared  = " + str(ldaRscore) + "\n\n")

def saveGnb():
    X = df[getFeatures()]
    y = df["class"]
    gnb = GaussianNB().fit(X, y)
    y_pred = gnb.predict(X)
    gnbRscore = gnb.score(X, y)
    f = open("./results/gnb.txt", "w")
    f.write("Misclassification  = " + str(calculateMisclassification(y, y_pred)) + "\n\n")
    f.write("R-squared  = " + str(gnbRscore) + "\n\n")

def calculateMisclassification(y, y_pred):
    y = y.values
    misclassificationSum = 0
    for i in range(len(y)):
        misclassificationSum +=1 if y[i] != y_pred[i] else 0
    misclassificationError =misclassificationSum /len(y_pred)
    return  round(misclassificationError, 4)

#### Phase1 of Assignment2
# saveLogisticRegression()
# saveQda()
# saveLda()
# saveGnb()


#### Phase2 of Assignment3
def saveLinearRegression():
    X = df[getFeatures()]
    y = df["class"]
    linearRegression = LinearRegression().fit(X,y)
    y_pred = linearRegression.predict(X)
    y_pred_classified = []
    for i in range(len(y_pred)):
        if(y_pred[i] >3):
            y_pred_classified.append(4)
        else:
            y_pred_classified.append(2)
    f = open("./results/LinearRegression.txt", "w")
    f.write("Misclassification  = " + str(calculateMisclassification(y, y_pred_classified)) + "\n\n")
    f.write("R-squared  = " + str(linearRegression.score(X,y)) + "\n\n")
saveLinearRegression()