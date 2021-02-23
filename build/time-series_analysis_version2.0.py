#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import stats
import matplotlib.backends.backend_pdf
import math
import random
from matplotlib import pyplot as plt 
import numpy as np  
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import sys
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.charts import Bar
from pyecharts import options as opts
import dataframe_image as dfi
from jupyterthemes import get_themes
import jupyterthemes as jt
from jupyterthemes.stylefx import set_nb_theme
from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))
#!{sys.executable} -m pip install dataframe_image
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


df = pd.read_csv("data/" + "3" + ".csv",skiprows=1)

#     df['difference'] = df.iloc[:,1].diff()
#     df = df.iloc[1:]
df.head()
#Rename the columns
df.columns = ['date', 'value']
df.head()


# In[3]:


df.date = pd.to_datetime(df.date)
df.set_index('date', inplace=True)
df.head()


# In[ ]:





# In[10]:


df[['value']].plot(figsize = (20,10), linewidth = 5, fontsize = 20)
plt.xlabel('Date', fontsize = 30)
plt.ylabel('Load Value', fontsize = 30)
plt.title('Load Value Time Series', fontsize = 40)
plt.legend(loc=2, prop={'size': 20})


# In[11]:


print("Smoothing")
values = df[['value']]
values.rolling(14).mean().plot(figsize = (20,10), linewidth = 5, fontsize = 20)
plt.xlabel('Date', fontsize = 30)
plt.ylabel('Load Value', fontsize = 30)
plt.title('Smoothed out Time Series', fontsize = 40)
plt.legend(loc=2, prop={'size': 20})


# In[12]:


values.diff().plot(figsize = (20,10), linewidth = 5, fontsize = 20)
plt.xlabel('Date', fontsize = 30)
plt.ylabel('Load Value', fontsize = 30)
plt.title('Differenced Time Series', fontsize = 40)
plt.legend(loc=2, prop={'size': 20})


# In[16]:


df.corr()


# In[ ]:





# In[17]:


values = df['value']
pd.plotting.autocorrelation_plot(values)


# In[ ]:




