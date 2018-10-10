import numpy as np
import pandas as pd
import statsmodels.api as sm
df = pd.read_csv("./data/breastData.csv", sep='\s*,\s*',
                           header=0, encoding='ascii', engine='python')
def regressionData(X,y):
    X = sm.add_constant(X)
    results = sm.OLS(y,X).fit()
    return results.summary()
    
for x in range(1,len(df.columns.tolist())-1):
    print(df.columns.tolist()[x])
    X= np.array(df[df.columns.tolist()[x]]).reshape(-1,1)
    y= np.array(df["class"]).reshape(-1,1)
    f = open("./results/"+df.columns.tolist()[x]+".csv", "w")
    f.write( str(regressionData(X, y)))

