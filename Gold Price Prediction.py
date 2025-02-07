#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf


# In[2]:


import requests
import pandas as pd
from bs4 import BeautifulSoup


# In[3]:


import seaborn as sns


# In[4]:


import pandas as pd


# In[5]:


import numpy as np


# In[6]:


import gradio as gr


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


from sklearn.model_selection import RandomizedSearchCV


# In[9]:


usd_inr = yf.download('USDINR=X', start='2024-01-01', end='2024-12-31', interval='1wk')


# In[10]:


type(usd_inr)


# In[11]:


usd_inr.head()


# In[12]:


usd_inr.info()


# In[13]:


usd_inr.reset_index(inplace=True)


# In[14]:


usd_inr=usd_inr[['Date','Close']]
usd_inr.columns = ['Date','USD_INR']


# In[15]:


usd_inr.head()


# In[16]:


import pandas as pd
gold_dataset = pd.read_csv("Gold vs USDINR.csv")


# In[17]:


gold_dataset.info()


# In[18]:


gold_dataset.head()


# In[19]:


gold_dataset['Goldrate'] = gold_dataset['Goldrate'].replace('₹', '', regex=True).replace(',','', regex=True).astype(float)


# In[20]:


sns.boxplot(gold_dataset['USD_INR'])


# In[21]:


gold_dataset['USD_INR'].min()


# In[22]:


sns.regplot(x='USD_INR', y='Goldrate', data=gold_dataset)


# In[23]:


X = gold_dataset[['USD_INR']]
y = gold_dataset[['Goldrate']]


# In[24]:


X


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)


# In[26]:


X_train.shape, X_test.shape


# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[28]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[29]:


X_test_scaled


# In[30]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[31]:


regressor.fit(X_train_scaled, y_train)


# In[32]:


regressor.get_params()


# In[33]:


regressor.coef_


# In[34]:


regressor.intercept_


# In[35]:


# y = mx+b
m = regressor.coef_[0][0]
b = regressor.intercept_[0]


# In[36]:


m,b


# In[37]:


x_train_predict = regressor.predict(X_train_scaled)


# In[38]:


plt.scatter(X_train,y_train)
plt.plot(X_train, x_train_predict, color='r')
plt.xlabel("USD_INR")
plt.ylabel("Goldrate")

plt.show()


# In[39]:


X_test_predicted = regressor.predict(X_test_scaled)


# In[40]:


X_test_predicted


# In[41]:


y_test


# In[42]:


from sklearn.metrics import mean_squared_error


# In[43]:


mean_squared_error(y_test, X_test_predicted)


# In[44]:


from sklearn.model_selection import RandomizedSearchCV
param_space = {'copy_X': [True,False], 
               'fit_intercept': [True,False], 
               'n_jobs': [1,5,10,15,None], 
               'positive': [True,False]}


# In[45]:


search = RandomizedSearchCV(regressor, param_space, n_iter=50, cv=5)


# In[46]:


search.fit(X_train_scaled, y_train)


# In[47]:


search.best_params_


# In[48]:


tuned_model = LinearRegression(positive= True, n_jobs= 1, fit_intercept= True, copy_X= True)


# In[49]:


tuned_model.fit(X_train_scaled, y_train)


# In[50]:


tuned_model.coef_


# In[51]:


tuned_model.intercept_


# In[52]:


import pickle


# In[53]:


#pickle.dump(regressor,open('regressor.pkl','wb'))


# In[54]:


regressor_reloaded = pickle.load(open('regressor.pkl','rb'))


# In[55]:


regressor_reloaded.coef_


# In[56]:


#pickle.dump(scaler,open('scaler.pkl','wb'))


# In[57]:


scaler=pickle.load(open('scaler.pkl','rb'))


# In[58]:


def calculate_gold_rate(usd_inr):
    scaled_input = scaler.transform(np.array(usd_inr).reshape(1,-1))
    return round(regressor.predict(scaled_input)[0][0],2)


# In[59]:


calculate_gold_rate(80)


# In[60]:


import gradio as gr

demo = gr.Interface(
    fn=calculate_gold_rate,
    inputs=["number"],
    outputs=["number"],
    title="How much is 1g gold now?"
)

demo.launch()


# In[ ]:





# In[ ]:




