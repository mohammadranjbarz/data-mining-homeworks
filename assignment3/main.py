import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from markdown import markdown
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("./breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def get_features():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    return features





def generate_readme_html():
    input_filename = 'Readme.md'
    output_filename = 'Readme.html'

    f = open(input_filename, 'r')
    html_text = markdown(f.read(), output_format='html4')
    file = open(output_filename, "w")
    file.write(str(html_text))
generate_readme_html()

X = df[get_features()]
y = df["class"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
path1 = regr_1.decision_path(X_test)
print(path1)
print(regr_1.score(X_test,y_1))
print(regr_2.score(X_test,y_1))


from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


dot_data = StringIO()
export_graphviz(path1, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.draw('file.png')
graph.write_png("./dtree.png")

