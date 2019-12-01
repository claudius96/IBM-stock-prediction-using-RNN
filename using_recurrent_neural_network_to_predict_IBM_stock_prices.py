#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

#importing the dataset
dataset = pd.read_csv('IBM_2006-01-01_to_2018-01-01.csv')
training_set = dataset[0:2000].iloc[:,1:2].values

#scaling the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_set = scaler.fit_transform(training_set)

X_train = []
y_train = []

for i in range(60,len(scaled_training_set)):
    X_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

reg = Sequential()
#first layer
reg.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
reg.add(Dropout(0.2))
#second layer
reg.add(LSTM(units=50,return_sequences=True))
reg.add(Dropout(0.2))
#third layer
reg.add(LSTM(units=50))
reg.add(Dropout(0.2))
#adding the dense layer
reg.add(Dense(units=1))

#compiling the RNN
reg.compile(optimizer='adam', loss='mean_squared_error')
reg.fit(X_train, y_train, batch_size=32, epochs=40)

#fitting the model to the test set
test_set = dataset[2000:].iloc[:,1:2].values
real_data = dataset.iloc[:,1:2].values
inputs = real_data[len(real_data) -len(test_set)-60:]
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,len(inputs)):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))
predicted_stock = reg.predict(X_test)
predicted_stock = scaler.inverse_transform(predicted_stock)


# Visualising the results
real = real_data[len(real_data) -len(test_set)-60:]
plt.plot(real, color = 'red', label = 'Real IBM stock price')
plt.plot(predicted_stock, color = 'blue', label = 'Predicted IBM Stock Price')
plt.title('IBM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IBM Stock Price')
plt.legend()
plt.show()






    


    


