import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
from keras.layers import Input, Dense
from keras.models import Model,Sequential

df = pd.read_csv("./pima-indians-diabetes.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def get_features():
    features = df.columns.tolist()
    del features[8]
    return features
data = np.array(df)
# data = data.reshape((data.shape[0],data.shape[1],1))
print(data.shape)
print(df.iloc[[]])

from numpy import genfromtxt
my_data = genfromtxt('./pima-indians-diabetes.csv', delimiter=',')
print(my_data[3])
print(my_data.shape)

input_data=[]
output_data = []
for i in range(my_data.shape[0]):
    y = my_data[i]
    last, y = y[-1], y[:-1]
    output_data.append(y)
    input_data.append(last)
input_data = np.array(input_data)
output_data = np.array(output_data)


# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

#
#
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 3  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
#
# # this is our input placeholder
input_img = Input(shape=(9,))

# # "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# # "decoded" is the lossy reconstruction of the input
decoded = Dense(9, activation='sigmoid')(encoded)
# # this model maps an input to its reconstr
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
print('np.prod(x_train.shape[1:]) ', np.prod(x_train.shape[1:]))
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
#
# print(x_train.shape)
# print(x_test.shape)
#
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))