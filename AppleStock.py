#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:/Data/all_apple_stock_data.csv")
df.head()


# In[3]:


df.columns


# In[4]:


df1 = df[["Open"]].values
df1 


# In[5]:


df1 = df['Open'].replace({'\$':'',',':''}, #replacing the $ signs
                        regex=True).astype(float)
df1


# In[6]:


print(df1.shape)


# In[7]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1));
df1


# In[8]:


num_of_prices = len(df1) #number of opening prices of the stock
print(len(df1))


# In[9]:


apple_price_features = [] #feature group input values-xs. we are going to fit our xs(independent variables)
apple_price_labels = [] # label group-output values-ys. we are going to fit our y(dependent variable)
for i in range(10, (num_of_prices)): 
    apple_price_features.append(df1[i-10:i, 0]) #the feature values between 0 and 9 of df1 scaled, grouping
#makes a group of 10
    apple_price_labels.append(df1[i,0]) #1 number for 10 inputs or 1 output for 10 inputs
#produce only one value for y--> one value each for 10 prices.


# In[10]:


x_train = np.array(apple_price_features[0:100]) #taking the first 100 features
y_train = np.array(apple_price_labels[0:100])


# In[11]:


print(x_train.shape)
print(y_train.shape)


# In[12]:


x_test = np.array(apple_price_features[101:]) #going from 101 until the end of the prices
y_test = np.array(apple_price_labels[101:])


# In[13]:


x_test = np.array(apple_price_features[101:]) #corrected slicing
y_test = np.array(apple_price_labels[101:])
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test.shape)


# In[14]:


from tensorflow.keras.layers import Input, Dense, Dropout, LSTM 
#keras has the neural network models--> importing one such model
from tensorflow.keras.models import Model


# In[15]:


input_layer = Input(shape = (x_train.shape[1], 1))
lstm1 = LSTM(100, activation = 'relu', return_sequences = True)(input_layer) #relu only classifies positive values
drop1 = Dropout(0.1)(lstm1)
lstm2 = LSTM(120, activation = 'relu', return_sequences = True)(drop1)
lstm3 = LSTM(80, activation = 'relu', return_sequences = True)(lstm2)
lstm4 = LSTM(50, activation = 'relu')(lstm3)
output_layer = Dense(1)(lstm4)
model = Model(input_layer, output_layer)
model.compile(optimizer = 'adam', loss = 'mse')
print(model.summary())
print(x_train.shape)
print(y_train.shape)


# In[16]:


model_history = model.fit(x_train, y_train, batch_size = 20, epochs = 100, validation_data =(x_test, y_test), verbose = 1,)


# In[19]:


y_pred = model.predict(x_test)
print(y_pred.shape)


# In[20]:


y_pred = scaler.inverse_transform(y_pred)
print(y_pred)


# In[21]:


print(y_test.shape)


# In[22]:


y_test = y_test.reshape(-1,1)
y_test = scaler.inverse_transform(y_test)
print(y_test)


# In[23]:


plt.figure(figsize=(6,5))
plt.plot(y_test, color = 'red', label = 'Historical Apple Stock Price')
plt.plot(y_pred, color = 'green', label = 'Predicted Apple Stock Price')
plt.title("Apple Stock Prices")
plt.xlabel("Date")
plt.legend()
plt.show()


# In[24]:


new_array = np.array([142, 141, 140, 144, 146, 148, 149, 149, 148])
new_array = new_array.reshape(-1, 1)
print(new_array)
print(new_array.shape)


# In[25]:


new_array_scaled = scaler.fit_transform(new_array)
new_array_scaled = np.reshape(new_array_scaled, (1,9,1))
print(new_array_scaled.shape)
print(new_array_scaled)


# In[26]:


new_pred = model.predict(new_array_scaled)


# In[27]:


new_pred = scaler.inverse_transform(new_pred)
print("We predict the price of Apple Stock will be:")
print(*new_pred[0])


# In[ ]:




