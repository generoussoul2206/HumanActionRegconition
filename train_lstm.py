import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
bodyswing_df = pd.read_csv("BODYSWING.txt")
lefthandswing_df = pd.read_csv("LEFTHANDSWING.txt")
rightheadswing_df = pd.read_csv("RIGHTHANDSWING.txt")
X = []
Y = []
no_of_timesteps = 10

# lấy dữ liệu để train
dataset = bodyswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    Y.append([1,0,0]) # bodyswing : 0
# --

dataset = lefthandswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    Y.append([0,1,0]) # lefthandswing : 1

dataset = rightheadswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    Y.append([0,0,1]) # righthandswing : 2

X,Y = np.array(X), np.array(Y)

print(X.shape, Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=50,return_sequences= True ,input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2)) # Loại bỏ ngẫu nhiên 1 số neural trong mạng
model.add(LSTM(units=50,return_sequences= True ))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences= True ))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer="adam", metrics=['accuracy'], loss="binary_crossentropy")

model.fit(X_train, Y_train, epochs=16, batch_size=32, validation_data=(X_test, Y_test))
model.save("model.h5")