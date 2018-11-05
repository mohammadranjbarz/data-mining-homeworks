import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from markdown import markdown

df = pd.read_csv("./breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def regression_data(X, y, featureName):
    # plt.scatter(X, y, color='black')
    plt.plot(X, y, 'ro')
    plt.axis()
    # plt.xticks(())
    # plt.yticks(())

    # plt.show()
    plt.savefig(f"./results/plots/{featureName}.png")
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    return results.summary()


# Call this function to save the summary of all features regression with y and save the data in results folder
def save_all_linear_regressions():
    for x in range(1, len(df.columns.tolist()) - 1):
        print(df.columns.tolist()[x])
        X = df[df.columns.tolist()[x]]
        y = df["class"]
        f = open("./results/" + df.columns.tolist()[x] + ".txt", "w")
        f.write(str(regression_data(X, y, df.columns.tolist()[x])))


# Call this function to save the summary of all features multipleRegression result in allFeatures.txt
def save_multiple_linear_regression_for_all_features():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    X = df[features]
    y = df["class"]

    rss = get_rss(X, y)
    f = open("./results/allFeatures.txt", "w")
    f.write("\nRss :"+str(rss) +"\n\n"+ str(regression_data(X, y,"allFeatures")) )


def get_rss(X, y):
    linearRegression = LinearRegression()
    linearRegression.fit(X, y)
    predictions = linearRegression.predict(X)
    RSS = sum((predictions - y) ** 2)
    return RSS


# Call this function to save the summary of significant features multipleRegression result in allFeatures.txt
def save_multiple_linear_regression_for_all_significant_features():
    features = ["clumpThickness", "uniformityOfCellSize", "bareNuclei", "blandChromatin",
                "normalNucleoli", "uniformityOfCellSize"]
    X = df[features]
    y = df["class"]
    rss = get_rss(X, y)
    f = open("./results/allSignificantFeatures.txt", "w")
    f.write("\nRss :"+str(rss) +"\n\n"+str(regression_data(X, y,"allSignificantFeatures")))


def save_ridge_regression():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    X = df[features]
    y = df["class"]
    baseAlpha = 0.1
    Ridgemodel = Ridge(alpha=baseAlpha)
    Ridgemodel.fit(X, y)
    baseScore = Ridgemodel.score(X, y, sample_weight=None)

    for x in range(1, 1000):
        alpha = 0.1 * x
        Ridgemodel = Ridge(alpha=alpha)
        Ridgemodel.fit(X, y)
        if (Ridgemodel.score(X, y, sample_weight=None) > baseScore):
            baseAlpha = alpha
            baseScore = Ridgemodel.score(X, y, sample_weight=None)

    Ridgemodel = Ridge(alpha=baseAlpha)
    Ridgemodel.fit(X, y)
    params = np.append(Ridgemodel.intercept_, Ridgemodel.coef_)
    predictions = Ridgemodel.predict(X)
    myDF3 = get_formatted_data_frame_from_predictions(X, y, predictions, params, features)
    f = open("./results/ridgeRegression.txt", "w")
    f.write("Alpha  = " + str(baseAlpha) + "\n\n")
    f.write("R-squared  = " + str(Ridgemodel.score(X, y, sample_weight=None)) + "\n\n")
    f.write(str(myDF3))


def save_Lasso_regression():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    X = df[features]
    y = df["class"]
    baseAlpha = 0.1
    LassoModel = Lasso(alpha=baseAlpha)
    LassoModel.fit(X, y)
    baseScore = LassoModel.score(X, y, sample_weight=None)

    for x in range(1, 1000):
        alpha = 0.1 * x
        LassoModel = Lasso(alpha=alpha)
        LassoModel.fit(X, y)
        if (LassoModel.score(X, y, sample_weight=None) > baseScore):
            baseAlpha = alpha
            baseScore = LassoModel.score(X, y, sample_weight=None)

    LassoModel = Lasso(alpha=baseAlpha)
    LassoModel.fit(X, y)
    params = np.append(LassoModel.intercept_, LassoModel.coef_)
    predictions = LassoModel.predict(X)
    myDF3 = get_formatted_data_frame_from_predictions(X, y, predictions, params, features)
    f = open("./results/lassoRegression.txt", "w")
    f.write("Alpha  = " + str(baseAlpha) + "\n\n")
    f.write("R-squared  = " + str(LassoModel.score(X, y, sample_weight=None)) + "\n\n")
    f.write(str(myDF3))


def save_elastic_net_regression():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    X = df[features]
    y = df["class"]
    baseAlpha = 0.1
    ElNet = ElasticNet(random_state=0, alpha=baseAlpha)
    ElNet.fit(X, y)
    baseScore = ElNet.score(X, y, sample_weight=None)

    for x in range(1, 1000):
        alpha = 0.1 * x
        ElNet = ElasticNet(random_state=0, alpha=alpha)
        ElNet.fit(X, y)
        # print("alpha  : ", str(alpha), ", score:  "
        #       , str(ElNet.score(X, y, sample_weight=None)))
        if (ElNet.score(X, y, sample_weight=None) > baseScore):
            baseAlpha = alpha
            baseScore = ElNet.score(X, y, sample_weight=None)

    ElNet = ElasticNet(random_state=0, alpha=baseAlpha)
    ElNet.fit(X, y)
    params = np.append(ElNet.intercept_, ElNet.coef_)
    predictions = ElNet.predict(X)
    params = np.round(params, 4)
    myDF3 = get_formatted_data_frame_from_predictions(X, y, predictions, params, features)
    f = open("./results/elasticNetRegression.txt", "w")
    f.write("Alpha  = " + str(baseAlpha) + "\n\n")
    f.write("R-squared  = " + str(ElNet.score(X, y, sample_weight=None)) + "\n\n")
    f.write(str(myDF3))


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
    return myDF3


def generate_readme_html():
    input_filename = 'Readme.md'
    output_filename = 'Readme.html'

    f = open(input_filename, 'r')
    html_text = markdown(f.read(), output_format='html4')
    file = open(output_filename, "w")
    file.write(str(html_text))


# generate_readme_html()
# save_all_linear_regressions()
# save_multiple_linear_regression_for_all_features()
# save_multiple_linear_regression_for_all_significant_features()
# save_elastic_net_regression()
# save_ridge_regression()
save_Lasso_regression()
