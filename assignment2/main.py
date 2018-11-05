import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats



df = pd.read_csv("./breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')

def getFeatures():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    return features

def get_formatted_data_frame_from_predictions(X, y, predictions, params, features):
    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))
    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)
    myDF3 = pd.DataFrame()
    features.insert(0, "constants")
    myDF3["Feature"], myDF3["Coefficients"], myDF3["t values"], myDF3["Standard Errors"], myDF3["Probabilites"] = [
        features, params, ts_b, sd_b, p_values]
    return MSE, myDF3

def saveLogisticRegression():
    X = df[getFeatures()]
    y = df["class"]
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    y_pred = logreg.predict(X)
    logesticRscore = logreg.score(X, y)
    params = np.append(logreg.intercept_, logreg.coef_)
    MSE, result = get_formatted_data_frame_from_predictions(X, y, y_pred, params, getFeatures())
    print(MSE)
    print(result)
    f = open("./results/LogisticRegression.txt", "w")
    f.write("MSE  = " + str(MSE) + "\n\n")
    f.write("R-squared  = " + str(logesticRscore) + "\n\n")
    f.write(str(result))

saveLogisticRegression()