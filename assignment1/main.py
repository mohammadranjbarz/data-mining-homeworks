import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("./data/breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')

def regressionData(X, y):
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    return results.summary()


# Call this function to save the summary of all features regression with y and save the data in results folder
def saveAllLinearRegressions():
    for x in range(1, len(df.columns.tolist()) - 1):
        print(df.columns.tolist()[x])
        X = df[df.columns.tolist()[x]]
        y = df["class"]
        f = open("./results/" + df.columns.tolist()[x] + ".txt", "w")
        f.write(str(regressionData(X, y)))


# Call this function to save the summary of all features multipleRegression result in allFeatures.txt
def saveMultipleLinearRegressionForAllFeatures():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    X = df[features]
    y = df["class"]
    f = open("./results/allFeatures.txt", "w")
    f.write(str(regressionData(X, y)))