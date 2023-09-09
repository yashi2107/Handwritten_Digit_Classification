#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow


# In[5]:


from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


# In[6]:


(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()


# In[7]:


x_train


# In[13]:


x_train[0].shape


# In[15]:


x_test.shape


# In[16]:


y_train


# In[18]:


import matplotlib.pyplot as plt
plt.imshow(x_train[1])


# In[19]:


x_train = x_train/255
x_test = x_test/255


# In[20]:


x_train[0]


# In[44]:


model= Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10, activation = 'softmax'))


# In[45]:


model.summary()


# In[46]:


model.compile(loss='sparse_categorical_crossentropy', optimizer = 'Adam',metrics = ['accuracy'])


# In[47]:


history = model.fit(x_train, y_train, epochs = 50, validation_split=0.2)


# In[48]:


y_prob = model.predict(x_test)


# In[49]:


y_pred = y_prob.argmax(axis=1)


# In[50]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[51]:


plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 


# In[52]:


plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 


# In[60]:


plt.imshow(x_test[0])


# In[61]:


model.predict(x_test[0].reshape(1,28,28)).argmax(axis=1)


# In[ ]:




