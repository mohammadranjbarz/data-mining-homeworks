import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from markdown import markdown
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_squared_error

df = pd.read_csv("./breastData.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def get_features():
    features = df.columns.tolist()
    del features[10]
    del features[0]
    return features


def get_model_report(y, y_pred):
    # mis_classification = calculate_mis_classification(y, y_pred)
    # f1 = f1_score(y, y_pred, average="macro")
    # precision = precision_score(y, y_pred, average="macro")
    # recall = recall_score(y, y_pred, average="macro")
    # accuracy = accuracy_score(y, y_pred)
    # return format_result_scores(accuracy, f1, mis_classification, precision, recall)
    return get_model_report_for_multi_y_pred([y], [y_pred])


def get_model_report_for_multi_y_pred(y_list, y_pred_list):
    mis_classification = 0
    f1 = 0
    precision = 0
    recall = 0
    accuracy = 0
    r2 = 0
    mse = 0
    for i in range(len(y_pred_list)):
        y_pred = y_pred_list[i]
        y = y_list[i]
        mis_classification += calculate_mis_classification(y, y_pred)
        f1 += f1_score(y, y_pred, average="macro")
        precision += precision_score(y, y_pred, average="macro")
        recall += recall_score(y, y_pred, average="macro")
        accuracy += accuracy_score(y, y_pred)
        r2 += r2_score(y, y_pred)
        mse += mean_squared_error(y, y_pred)

    return format_result_scores(accuracy / len(y_pred_list), f1 / len(y_pred_list)
                                , mis_classification / len(y_pred_list),
                                precision / len(y_pred_list), recall / len(y_pred_list),r2 / len(y_pred_list) ,mse / len(y_pred_list) )


def format_result_scores(accuracy, f1, mis_classification, precision, recall, r2, mse):
    return "MisClassification  = " + str(round(mis_classification, 4)) + "\n\n" + "Accuracy  = " + str(
        round(accuracy, 4)) + "\n\n" + "F1 score  =  " + str(
        round(f1, 4)) + "\n\n" + "Precision score  =  " + str(
        round(precision, 4)) + "\n\n" + "Recall score  =  " + str(
        round(recall, 4)) + "\n\n" +"r2   =  "+str(
        round(r2, 4)) + "\n\n" +"mse   =  "+str(
        round(mse,4))


def save_logistic_regression(X, y):
    logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    y_pred = logreg.predict(X)
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


#### Phase2 of Assignment2
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


# save_linear_regression(X, y)


##### Phase3 of Assignment2
def save_regression_k_fold(X, y, classification_model, classification_model_name):
    k_folde_number = 5
    y_preds_list = []
    y_list = []
    kf = KFold(n_splits=k_folde_number, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressionFunction = classification_model.fit(X_train, y_train)
        y_pred = regressionFunction.predict(X_test)
        y_list.append(y_test)
        y_preds_list.append(y_pred)

    f = open(f"./results/k_fold/{classification_model_name}.txt", "w")
    f.write(get_model_report_for_multi_y_pred(y_list, y_preds_list))


def save_regression_leave_one_out(X, y, classification_model, classification_model_name):
    y_preds_list = []
    y_list = []
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressionFunction = classification_model.fit(X_train, y_train)
        y_pred = regressionFunction.predict(X_test)
        y_list.append(y_test)
        y_preds_list.append(y_pred)

    f = open(f"./results/leave_one_out/{classification_model_name}.txt", "w")
    f.write(get_model_report_for_multi_y_pred(y_list, y_preds_list))


###### k-fold

def save_logistic_regression_k_fold(X, y):
    save_regression_k_fold(X, y,
                           LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                           "LogisticRegression")


def save_qda_k_fold(X, y):
    qda = QDA()
    save_regression_k_fold(X, y, qda, "Qda")


def save_lda_k_fold(X, y):
    lda = LDA()
    save_regression_k_fold(X, y, lda, "Lda")


def save_gnb_k_fold(X, y):
    gnb = GaussianNB()
    save_regression_k_fold(X, y, gnb, "gnb")


save_logistic_regression_k_fold(X, y)
save_gnb_k_fold(X, y)
save_lda_k_fold(X, y)
save_qda_k_fold(X, y)


###### leave_one_out

def save_logistic_regression_leave_one_out(X, y):
    save_regression_leave_one_out(X, y,
                                  LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                                  "LogisticRegression")


def save_qda_leave_one_out(X, y):
    qda = QDA()
    save_regression_leave_one_out(X, y, qda, "Qda")


def save_lda_leave_one_out(X, y):
    lda = LDA()
    save_regression_leave_one_out(X, y, lda, "Lda")


def save_gnb_leave_one_out(X, y):
    gnb = GaussianNB()
    save_regression_leave_one_out(X, y, gnb, "gnb")


save_logistic_regression_leave_one_out(X, y)
save_gnb_leave_one_out(X, y)
save_lda_leave_one_out(X, y)
save_qda_leave_one_out(X, y)




def generate_readme_html():
    input_filename = 'Readme.md'
    output_filename = 'Readme.html'

    f = open(input_filename, 'r')
    html_text = markdown(f.read(), output_format='html4')
    file = open(output_filename, "w")
    file.write(str(html_text))
# generate_readme_html()