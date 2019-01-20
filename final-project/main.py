import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

from numpy import genfromtxt
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
my_data = genfromtxt('./pima-indians-diabetes.csv', delimiter=',')

input_data = []
output_data = []
for i in range(my_data.shape[0]):
    y = my_data[i]
    last, y = y[-1], y[:-1]
    input_data.append(y)
    output_data.append([last])
input_data = np.array(input_data)
output_data = np.array(output_data)

from keras.layers import Input, Dense, Activation
from keras.models import Sequential
from sklearn.decomposition import PCA

# calculateKerasModel()
# stackedAutoEncoder()


def format_result_scores(accuracy, mis_classification, f1, precision, recall, r2, mse, confussion_matrix):
    return "MisClassification  = " + str(round(mis_classification, 4)) + "\n" + "Accuracy  = " + str(
        round(accuracy, 4)) + "\n" + "F1 score  =  " + str(
        round(f1, 4)) + "\n" + "Precision score  =  " + str(
        round(precision, 4)) + "\n" + "Recall score  =  " + str(
        round(recall, 4)) + "\n" + "r2   =  " + str(
        round(r2, 4)) + "\n" + "confusion matrix   =  " + str(
        confussion_matrix) + "\n" + "mse   =  " + str(
        round(mse, 4))


def get_model_result(y, y_pred):
    return format_result_scores(accuracy_score(y, y_pred), 1 - accuracy_score(y, y_pred),
                                f1_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred),
                                r2_score(y, y_pred),
                                mean_squared_error(y, y_pred), confusion_matrix(y, y_pred))


def get_logistic_regression(x_train, y_train, x_test, y_test):
    logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    return get_model_result(y_test, y_pred)


def get_qda_report(x_train, y_train, x_test, y_test):
    qda = QDA().fit(x_train, y_train)
    y_pred = qda.predict(x_test)
    return get_model_result(y_test, y_pred)


def get_lda_report(x_train, y_train, x_test, y_test):
    lda = LDA().fit(x_train, y_train)
    y_pred = lda.predict(x_test)
    return get_model_result(y_test, y_pred)


def get_gnb_report(x_train, y_train, x_test, y_test):
    gnb = GaussianNB().fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    return get_model_result(y_test, y_pred)


def get_svm(x_train, y_train, x_test, y_test):
    svmachine = svm.SVC(gamma='auto', kernel='linear')
    svm_model = svmachine.fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)
    return get_model_result(y_test, y_pred)


def get_random_forrest(x_train, y_train, x_test, y_test):
    random_forrest = RandomForestClassifier()
    model = random_forrest.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return get_model_result(y_test, y_pred)


def get_bagging(x_train, y_train, x_test, y_test):
    bagging = BaggingClassifier()
    bagging_model = bagging.fit(x_train, y_train)
    y_pred = bagging_model.predict(x_test)
    return get_model_result(y_test, y_pred)


def get_decision_tree(x_train, y_train, x_test, y_test):
    model = tree.DecisionTreeRegressor(max_depth=4)
    tree_model = model.fit(x_train, y_train)
    y_pred = tree_model.predict(y_train)
    return get_model_result(y_test, y_pred)


def get_scores_for_regression_tree(x_train, y_train, x_test, y_test):
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_1.fit(x_train, y_train)
    y_pred = regr_1.predict(x_test)
    return get_model_result(y_test, y_pred)


def calculateKerasModel(x_train, y_train, x_test,y_test):
    encoding_dim = 3  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    input_img = Input(shape=(8,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(8, activation='sigmoid')(encoded)
    print(x_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    #
    model = Sequential()
    model.add(Dense(1, input_dim=8))
    model.add(Activation('relu'))
    print('int(x_train.shape[0])', int(x_train.shape[0] / 10))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=int(x_train.shape[0] / 10), validation_data=(x_test, y_test))
    model.predict(x_train)


def stackedAutoEncoder(x_train, y_train, x_test,y_test):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    import numpy as np

    data_dim = 8

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(data_dim, return_sequences=True,
                   input_shape=8))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=64, epochs=5,
              validation_data=(x_test, y_test))




def calculate_result_normal_inputs():
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)
    result = ""
    result += "logistic regression :\n" + get_logistic_regression(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "lda :\n" + get_lda_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    # result += "qda :\n" + get_qda_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "gnb :\n" + get_gnb_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "svm :\n" + get_svm(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    # result += "random forrest :\n" + get_random_forrest(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "bagging :\n" + get_bagging(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "decision tree :\n" + get_bagging(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    f = open("./results/normal_data.txt", "w")
    f.write(result)


def getPCA(x,n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x)
    # print(pca.explained_variance_ratio_)
    # print(pca.transform(x_train))
    # print(pca.fit_transform(x))
    return pca.fit_transform(x)

def calculate_result_normal_inputs():
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)
    result = ""
    result += "logistic regression :\n" + get_logistic_regression(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "lda :\n" + get_lda_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "qda :\n" + get_qda_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "gnb :\n" + get_gnb_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "svm :\n" + get_svm(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "random forrest :\n" + get_random_forrest(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "bagging :\n" + get_bagging(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "decision tree :\n" + get_bagging(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    f = open("./results/normal_data.txt", "w")
    f.write(result)

def calculate_result_pca_inputs(n_components):
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)
    x_train = getPCA(x_train,n_components)
    x_test= getPCA(x_test,n_components)
    result = ""
    result += "logistic regression :\n" + get_logistic_regression(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "lda :\n" + get_lda_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "qda :\n" + get_qda_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "gnb :\n" + get_gnb_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "svm :\n" + get_svm(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "random forrest :\n" + get_random_forrest(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "bagging :\n" + get_bagging(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "decision tree :\n" + get_bagging(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    f = open(f"./results/pca_data{n_components}.txt", "w")
    f.write(result)


# calculate_result_normal_inputs()
calculate_result_pca_inputs(1)
# x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

# print('predict : ',calculateKerasModel(x_train,y_train,x_test,y_test))