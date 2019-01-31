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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, roc_auc_score

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

from numpy import genfromtxt
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.DataConversionWarning)
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


def format_result_scores(roc_auc_score, accuracy, f1, precision, recall, r2, mse, confussion_matrix):
    return \
        "auc_score  = " + str(round(roc_auc_score, 4)) + \
        "\n" + "Accuracy  = " + str(
            round(accuracy, 4)) + "\n" + "F1 score  =  " + str(
            round(f1, 4)) + "\n" + "Precision score  =  " + str(
            round(precision, 4)) + "\n" + "Recall score  =  " + str(
            round(recall, 4)) + "\n" + "r2   =  " + str(
            round(r2, 4)) + "\n" + "confusion matrix   =  " + str(
            confussion_matrix) + "\n" + "mse   =  " + str(
            round(mse, 4))


def get_model_result(y, y_pred):
    return format_result_scores(roc_auc_score(y, y_pred), accuracy_score(y, y_pred),
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


def getPCA(x, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x)
    # print(pca.explained_variance_ratio_)
    # print(pca.transform(x_train))
    # print(pca.fit_transform(x))
    return pca.fit_transform(x)



def encodFunction_three_layers(x_train, x_test, first_layer_neurons, second_layer_neurons,
                               third_layer_nuerons):
    from keras.layers import Input, Dense
    from keras.models import Model
    input_x = Input(shape=(8,))

    encoded = Dense(first_layer_neurons, activation='relu')(input_x)
    encoded2 = Dense(second_layer_neurons, activation='relu')(encoded)
    encoded3 = Dense(third_layer_nuerons, activation='relu')(encoded2)
    decoded1 = Dense(second_layer_neurons, activation='sigmoid')(encoded3)
    decoded2 = Dense(first_layer_neurons, activation='sigmoid')(decoded1)

    decoded3 = Dense(8, activation='sigmoid')(decoded2)

    # this model maps an input to its reconstruction
    encoder = Model(input_x, encoded3)
    autoencoder = Model(input_x, decoded3)
    # this model maps an input to its encoded representation
    # encoder = Model(input_x, encoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=0,
                    batch_size=800,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    encoded_x_test = encoder.predict(x_test)
    encoded_x_train = encoder.predict(x_train)
    return encoded_x_train, encoded_x_test


def encodFunction_two_layers(x_train, x_test, first_layer_neurons, second_layer_neurons):
    from keras.layers import Input, Dense
    from keras.models import Model
    input_x = Input(shape=(8,))

    encoded = Dense(first_layer_neurons, activation='relu')(input_x)
    encoded2 = Dense(second_layer_neurons, activation='relu')(encoded)
    decoded1 = Dense(second_layer_neurons, activation='sigmoid')(encoded2)
    decoded2 = Dense(8, activation='sigmoid')(decoded1)

    # this model maps an input to its reconstruction
    encoder = Model(input_x, encoded2)
    autoencoder = Model(input_x, decoded2)
    # this model maps an input to its encoded representation
    # encoder = Model(input_x, encoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=0,
                    batch_size=800,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    encoded_x_test = encoder.predict(x_test)
    encoded_x_train = encoder.predict(x_train)
    return encoded_x_train, encoded_x_test


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
    x_train = getPCA(x_train, n_components)
    x_test = getPCA(x_test, n_components)
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


def calculate_result_auto_encoder_inputs(layer1_neurons, layer2_neurons, layer3_neurons=0):
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)
    if layer3_neurons != 0:
        print('before shape', x_train.shape, x_test.shape)
        label = f"three layer - {str(layer1_neurons)}, {layer2_neurons}, {layer3_neurons} neurons"
        x_train, x_test = encodFunction_three_layers(x_train, x_test, layer1_neurons, layer2_neurons, layer3_neurons)
        print('after shape', x_train.shape, x_test.shape)

    else:

        label = f"two layer - {str(layer1_neurons)}, {layer2_neurons} neurons"
        x_train, x_test = encodFunction_two_layers(x_train, x_test, layer1_neurons, layer2_neurons)

    result = ""
    result += "logistic regression :\n" + get_logistic_regression(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "lda :\n" + get_lda_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "qda :\n" + get_qda_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "gnb :\n" + get_gnb_report(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "svm :\n" + get_svm(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "random forrest :\n" + get_random_forrest(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "bagging :\n" + get_bagging(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    result += "decision tree :\n" + get_bagging(x_train, y_train, x_test, y_test) + "\n\n\n*******\n"
    f = open(f"./tests/auto_encoder{label}.txt", "w")
    f.write(result)

# calculate_result_normal_inputs()
# calculate_result_pca_inputs(1)
# calculate_result_pca_inputs(2)
# calculate_result_pca_inputs(3)
# calculate_result_pca_inputs(4)
# calculate_result_pca_inputs(5)
# calculate_result_pca_inputs(6)
# calculate_result_pca_inputs(7)
# calculate_result_auto_encoder_inputs(7, 6, 5)
# calculate_result_auto_encoder_inputs(7, 6, 4)
# calculate_result_auto_encoder_inputs(7, 6, 3)
# calculate_result_auto_encoder_inputs(7, 6, 2)
# calculate_result_auto_encoder_inputs(7, 5, 4)
# calculate_result_auto_encoder_inputs(7, 5, 3)
# calculate_result_auto_encoder_inputs(7, 5, 2)
# calculate_result_auto_encoder_inputs(7, 4, 3)
# calculate_result_auto_encoder_inputs(7, 4, 2)
# calculate_result_auto_encoder_inputs(6, 5, 4)
# calculate_result_auto_encoder_inputs(6, 5, 3)
# calculate_result_auto_encoder_inputs(6, 5, 2)
# calculate_result_auto_encoder_inputs(6, 4, 2)
# calculate_result_auto_encoder_inputs(6, 3, 2)
# calculate_result_auto_encoder_inputs(5, 4, 3)
# calculate_result_auto_encoder_inputs(5, 4, 3)
# calculate_result_auto_encoder_inputs(5, 4, 2)
# calculate_result_auto_encoder_inputs(5, 3, 2)
# calculate_result_auto_encoder_inputs(4, 3, 2)
# calculate_result_auto_encoder_inputs(7, 6)
# calculate_result_auto_encoder_inputs(7, 5)
# calculate_result_auto_encoder_inputs(7, 4)
# calculate_result_auto_encoder_inputs(7, 3)
# calculate_result_auto_encoder_inputs(7, 2)
# calculate_result_auto_encoder_inputs(6, 5)
# calculate_result_auto_encoder_inputs(6, 4)
# calculate_result_auto_encoder_inputs(6, 3)
# calculate_result_auto_encoder_inputs(5, 4)
# calculate_result_auto_encoder_inputs(5, 3)
# calculate_result_auto_encoder_inputs(5, 2)
# calculate_result_auto_encoder_inputs(4, 3)
# calculate_result_auto_encoder_inputs(4, 2)
# calculate_result_auto_encoder_inputs(3, 2)
