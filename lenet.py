import numpy as np
import os
import pandas as pd
import urllib.request

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils.np_utils import to_categorical
from zipfile import ZipFile



# I like to declare constants to avoid typos later on especially if they'd be used in more than one place
ReluActivation = 'relu'
SoftmaxActivation = 'softmax'
AdamOptimizer = 'adam'
CategoricalCrossEntropy = 'categorical_crossentropy'


def fetch_data_sets():
    path = './data'
    print('fetching test and train datasets from kaggle . . .')
    url = 'https://www.kaggle.com/c/3004/download-all'
    train_set_csv_path = os.path.join(path, 'train.csv')
    test_set_csv_path = os.path.join(path, 'test.csv')
    zipped_file_path = os.path.join(path, 'data.zip')
    if os.path.exists(train_set_csv_path) and os.path.exists(test_set_csv_path):
        return
    os.makedirs(path, exist_ok=True)
    urllib.request.urlretrieve(url, zipped_file_path)
    zf = ZipFile(zipped_file_path, 'r')
    zf.extractall(path=path)
    zf.close()


training_set = pd.read_csv('./data/train.csv')
test_set = pd.read_csv('./data/test.csv')

Y_train = training_set[['label']]

X_train = training_set.drop(training_set.columns[[0]], axis=1)
X_test = test_set

X_train = np.array(X_train)
X_test = np.array(X_test)

# Reshape the training and test set
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Padding the images by 2 pixels since in the paper input images were 32x32
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# Standardization
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
X_train = (X_train - mean_px) / std_px

Y_train = to_categorical(Y_train)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation=ReluActivation,
                 input_shape=(32, 32, 1)))  # The original paper used sinh/tanh activation
model.add(MaxPooling2D(pool_size=2, strides=2))  # The original paper used Mean/Average pooling
model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation=ReluActivation, input_shape=(14, 14, 6)))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Flatten())

# Now, the fully connected layers
model.add(Dense(units=120, activation=ReluActivation))
model.add(Dense(units=84, activation=ReluActivation))

# Adding the softmax which did not originally exist in the paper
model.add(Dense(units=10, activation=SoftmaxActivation))

model.compile(optimizer=AdamOptimizer, loss=CategoricalCrossEntropy, metrics=['accuracy'])

model.fit(X_train, Y_train, steps_per_epoch=10, epochs=42)

y_pred = model.predict(X_test)

labels = np.argmax(y_pred, axis=1)

index = np.arange(1, 28001)

labels = labels.reshape([len(labels), 1])
index = index.reshape([len(index), 1])

solution_data_frame = pd.DataFrame()
solution_data_frame['ImageId'] = index
solution_data_frame['Label'] = labels

solution_data_frame.to_csv('./data/solution.csv')
