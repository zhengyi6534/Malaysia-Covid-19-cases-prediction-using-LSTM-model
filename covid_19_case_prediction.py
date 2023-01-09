#%%
# import packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tensorflow.keras.layers import SimpleRNN,Dense, LSTM
from tensorflow.keras import Sequential, Input, layers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import datetime
import pickle
from keras.layers import Dropout
from keras import applications
from tensorflow import keras


#%% Step 1) Data Loading
CSV_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')
train_df = pd.read_csv(CSV_PATH)

# %%
# step 2) Data Inspection/Visualization
train_df.describe()
train_df.info()
train_df.isna().sum() # to check the number of NaNs 
#no NaN

#%% Convert ? and empty data into NaN
train_df['cases_new'] = pd.to_numeric(train_df['cases_new'],errors='coerce')
train_df.info()
train_df.isna().sum()

#%% show the plot
plt.figure(figsize=(10,10))
plt.plot(train_df['cases_new'])
plt.show()

# %% 
# Step 3) Data cleaning
# to replace NaNs using interpolation approach
train_df['cases_new'] = train_df['cases_new'].interpolate(method='polynomial', order=2)

#double confirm if training data still have NaNs
train_df.isna().sum()

# %% 
# Step 4 Features Selection
cases_new = train_df['cases_new'].values # only select 1 feature

#%%
# Step 5 Data Preprocessing
mms = MinMaxScaler()
train_df = mms.fit_transform(cases_new.reshape(-1,1))

# Data splitting
X = []
y= []
win_size = 30

for i in range(win_size, len(train_df)):
    X.append(train_df[i-win_size:i])
    y.append(train_df[i])

X=np.array(X)
y=np.array(y)

X_train,X_test,y_train,y_test= train_test_split(
    X,y,train_size=0.3,random_state=123,shuffle=True)


#%% Model Development
model = Sequential()
model.add(LSTM(64,return_sequences=True, input_shape=(X_train.shape[1:])))
model.add(Dropout(0.3))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

#%%
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
ts_callback = TensorBoard(log_dir=LOGS_PATH)
es_callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)
hist = model.fit(X_train,y_train,epochs=10,batch_size=64, callbacks = [es_callback, ts_callback])

#validation_data=(X_test,y_test) is not needed

# %% Model Analysis
# Load the test data
TEST_CSV_PATH = os.path.join(os.getcwd(),'cases_malaysia_test.csv')
test_df = pd.read_csv(TEST_CSV_PATH)

#%% 
#Check the test data
test_df.describe()
test_df.info()
test_df.isna().sum()
# There is 1 Nan in the test data

#%%
# Clean the test data
# to replace NaNs using interpolation approach
test_df['cases_new'] = test_df['cases_new'].interpolate(method='polynomial', order=2)
test_df.isna().sum()
#no more NaN in the test data


#%%
test_df = test_df['cases_new']
test_df = mms.transform(test_df[::,None])


concatenated = np.concatenate((train_df,test_df))

#%% show the concatenated plot
plt.figure()
plt.plot(concatenated)
plt.show()

# min max transformation
X_testtest = []
y_testtest = []

for i in range(win_size, len(concatenated)):
    X_testtest.append(concatenated[i-win_size:i])
    y_testtest.append(concatenated[i])

X_testtest = np.array(X_testtest) # to convert into array
y_testtest = np.array(y_testtest)

predicted = model.predict(X_testtest) # to predict the unseen dataset

#%% Visualise the actual and predicted cases

plt.figure()
plt.plot(predicted, color = 'red')
plt.plot(y_testtest, color = 'blue')
plt.legend(['Predicted', 'Actual'])
plt.xlabel('time')
plt.ylabel('New Covid19 cases')
plt.show()

#actual, predicted
print(mean_absolute_percentage_error(y_testtest, predicted))
print(mean_absolute_error(y_testtest,predicted))
print(mean_squared_error(y_testtest, predicted))


#%% Model Deployment
model.save('model.h5')
