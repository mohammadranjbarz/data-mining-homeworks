import numpy as np
from sklearn.model_selection import train_test_split

from numpy import genfromtxt

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

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)
from keras.layers import Input, Dense, Activation
from keras.models import Sequential
from sklearn.decomposition import PCA


def calculateKerasModel():
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


def stackedAutoEncoder():
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


def calculatePCA():
    pca = PCA(n_components=1)
    pca.fit(x_train)
    print(pca.explained_variance_ratio_)
    # print(pca.transform(x_train))
    print(pca.fit_transform(x_train))


calculatePCA()
# calculateKerasModel()
# stackedAutoEncoder()
