import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

df = pd.read_csv("./breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def get_features():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    return features


X = df[get_features()]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=0)

clf.fit(X_train, y_train)

predict = clf.predict(X_test)
auc_score = roc_auc_score(y_test, predict)
f = open(f"./results/mlp.txt", "w")
f.write(f"classification_report:\n\n{classification_report(y_test,predict)}\n\n \n"
        f"auc_score : {auc_score}")
