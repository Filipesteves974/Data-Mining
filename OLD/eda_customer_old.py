#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries



# Remember: library imports are ALWAYS at the top of the script, no exceptions!
import sqlite3
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from itertools import product
from scipy.stats import skewnorm

from datetime import datetime
from sklearn.impute import KNNImputer
#from ydata_profiling import ProfileReport

# for better resolution plots

# Setting seaborn style
sns.set()


# # Reading the Data

# In[3]:


df_customer = pd.read_csv('/Users/miguelcaramelo/Desktop/Data_Science/1_semestre/Data_Mining/Projeto/Data-Mining/DM_AIAI_CustomerDB (1).csv', sep = ',')


# # Metadata

# In[ ]:





# In[4]:


df_customer.head()


# In[5]:


df_customer.tail()


# # Initial Analysis

# In[6]:


# Checking shape of dataframe
df_customer.shape


# In[7]:


# Checking the info of the dataframe
df_customer.info()


# In[60]:


df_customer.describe().T


# In[9]:


df_customer.isna().sum()


# In[10]:


# replace "" by nans
df_customer.replace("", np.nan, inplace=True)

# count of missing values
df_customer.isna().sum()


# In[ ]:


df_customer.describe(include='all').T


# In[59]:


df_customer.duplicated().sum()


# # Categorical Variables' Absolute Frequencies

# In[12]:


# Creating a list with the names of the non metric features
non_metric_features = ['First Name', 'Last Name', 'Customer Name', 'Country', 'Province or State', 'City','Gender', 'Education','Location Code', 'Marital Status','LoyaltyStatus','EnrollmentType']
for i in non_metric_features:
    # using the unique() method to see unique values in each non metric feature
    print(df_customer[i].unique())


# In[13]:


#Plot ALL Non Numeric Variables' Absolute Counts in one figure
non_metric_features = ['Location Code','Gender', 'Education', 'Marital Status','LoyaltyStatus','EnrollmentType']
sns.set() ## Reset to darkgrid

# Setting seaborn style
sns.set_style("whitegrid")

# Setting seaborn context
sns.set_context("notebook")



## What do these do?
sp_rows = 2
sp_cols = ceil(len(non_metric_features) / sp_rows)


# Prepare figure. Create individual axes where each histogram will be placed
fig, axes = plt.subplots(sp_rows, 
                         sp_cols, 
                         figsize=(20, 11),
                         tight_layout=True
                        )

# Plot data
# Iterate across axes objects and associate each histogram (hint: use the ax.hist() instead of plt.hist()):
for ax, feat in zip(axes.flatten(), non_metric_features): # Notice the zip() function and flatten() method
    sns.countplot(x=df_customer[feat],order=df_customer[feat].value_counts().index,ax=ax)
# Layout
# Add a centered title to the figure:
title = "NonNumeric Variables' Absolute Counts"

plt.suptitle(title)
"""
if not os.path.exists(os.path.join('..', 'figures', 'eda')):
    # if the eda directory is not present then create it first
    os.makedirs(os.path.join('..', 'figures', 'eda'))


plt.savefig(os.path.join('..', 'figures', 'eda', 'numeric_variables_boxplots.png'), dpi=200)
"""

plt.show()


# ## Bivariate Categorical Distribution

# In[14]:


cat1 = 'Gender'
cat2 = 'Education'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


plt.show()


# In[15]:


cat1 = 'Marital Status'
cat2 = 'Education'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


plt.show()


# In[16]:


cat1 = 'Marital Status'
cat2 = 'Gender'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[17]:


cat1 = 'LoyaltyStatus'
cat2 = 'Gender'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[18]:


cat1 = 'LoyaltyStatus'
cat2 = 'Education'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[19]:


cat1 = 'LoyaltyStatus'
cat2 = 'Marital Status'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[20]:


cat1 = 'EnrollmentType'
cat2 = 'Gender'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[21]:


cat1 = 'EnrollmentType'
cat2 = 'Education'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[22]:


cat1 = 'EnrollmentType'
cat2 = 'Marital Status'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[23]:


cat1 = 'EnrollmentType'
cat2 = 'LoyaltyStatus'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[24]:


cat1 = 'Location Code'
cat2 = 'Gender'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[25]:


cat1 = 'Location Code'
cat2 = 'Education'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[26]:


cat1 = 'Location Code'
cat2 = 'Marital Status'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[27]:


cat1 = 'Location Code'
cat2 = 'LoyaltyStatus'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[28]:


cat1 = 'Location Code'
cat2 = 'EnrollmentType'

catpc_df = df_customer.groupby([cat2, cat1])[cat1].size().unstack()

catpc_df.plot.bar(stacked=True)


# In[41]:


numeric_features = ['Income', 'Customer Lifetime Value']
for i in numeric_features:
    plt.figure(figsize=(10,5))
    sns.histplot(x=df_customer[i])
    plt.title(f'Histogram of {i}')
    plt.show()

sns.scatterplot(data=df_customer, x='Income', y='Customer Lifetime Value', edgecolor='black')
plt.title('Income vs Customer Lifetime Value')
plt.show()
sns.boxplot(x = df_customer['Income'])
plt.show()
sns.boxplot(x = df_customer[df_customer['Income']> 0]['Income'])
plt.show()
sns.boxplot(x = df_customer['Customer Lifetime Value'])
plt.show()


# In[42]:


df_customer[(df_customer['Income'] == 0) & (df_customer['Education'] == "College")]


# In[43]:


import pandas as pd

# Create a new categorical flag
df_customer['Income_Class'] = df_customer['Income'].apply(
    lambda x: 'Zero Income' if x == 0 else 'Non-Zero Income'
)

# Quick check
df_customer['Income_Class'].value_counts()


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.countplot(
    data=df_customer,
    x='Education',
    hue='Income_Class',
    palette=['#FF6F59', '#004E98']  # contrasting palette
)
plt.title('Education Distribution by Income Class')
plt.xlabel('Education Level')
plt.ylabel('Number of Customers')
plt.legend(title='Income Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[45]:


df_customer["Province or State"].nunique()


# ## Type conversion

# In[65]:


df_customer["EnrollmentDateOpening"] = pd.to_datetime(df_customer["EnrollmentDateOpening"])
df_customer["CancellationDate"] = pd.to_datetime(df_customer["CancellationDate"], errors='coerce')


# ## NaN's

# In[62]:


df_customer.isna().sum()


# In[61]:


df_customer[df_customer['Customer Lifetime Value'].isna()]


# By analyzing the previous dataframe, we can conclude that the NaN values in Income occur in the same rows as the NaN values in Customer Lifetime Value. Since there are only 20 rows, we will drop them in a future analysis.

# In the next delivery, we will handle the NaN values in CancellationDate as shown below, since we interpret NaN as indicating that the customer is still active. Next, we will create a histogram to compare customers who have already canceled the service with those who are still with us. 

# In[ ]:


print(df_customer["EnrollmentDateOpening"].max())
print(df_customer["CancellationDate"].max())


# In[70]:


"""df_customer['CustomerTenure'] = (df_customer['CancellationDate'].fillna(pd.to_datetime("2021-12-30 00:00:00") + pd.Timedelta(days=1)) - df_customer['EnrollmentDateOpening'])
df_customer"""


# #fazer histograma com as customr tenure
# # % de clientes ativos vs cancelados

# ## Futuras Alterações

# In[54]:


df_customer["Loyalty#"].count()-df_customer["Loyalty#"].nunique()


# Since df_customer["Loyalty#"].nunique() is smaller than df_customer["Loyalty#"].count() and df_customer["Customer Name"].nunique() is equal to df_customer[""Customer Name""].count(), we have different people with the same Loyalty#. Since the number of times that this happen (164) is insignificant compared to the number of lines (16921) in the next delivery we will drop them and later we can pass Loyalty# to index. When we do that we can drop the 3 columns that are related with the customer name. In the df_customer["Country"] we have always the result "Canada" so we can drop that too.

# ## Extra Points

# In[ ]:


import plotly.express as px

fig = px.scatter_geo(
    df_customer,
    lat='Latitude',
    lon='Longitude',
    color='Customer Lifetime Value',
    hover_name='City',
    hover_data=['Province or State', 'Income', 'LoyaltyStatus'],
    projection='natural earth',
    color_continuous_scale='viridis',
    title='Customer Distribution Across Canada',
    scope='north america'  # foca a vista na América do Norte
)
fig.update_geos(
    center=dict(lat=(115 / 2), lon=(-190/2)),  # coordenadas médias do Canadá
    lataxis_range=[40, 75],                   # latitude range (ajustável)
    lonaxis_range=[-140, -50],                # longitude range (ajustável)
    showcountries=True, countrycolor="LightGray",
    showland=True, landcolor="whitesmoke",
    lakecolor="LightBlue",
    showocean=True, oceancolor="aliceblue"
)
fig.show()


# In[57]:


import plotly.express as px

# Paleta azul-dourado premium (AIAI style)
aiai_palette = [(0, "#004E98"), (0.5, "#FFD23F"), (1, "#FF6F59")]

fig = px.scatter_geo(
    df_customer.dropna(subset=["Customer Lifetime Value"]),
    lat='Latitude',
    lon='Longitude',
    color='Customer Lifetime Value',
    hover_name='City',
    hover_data={
        'Province or State': True,
        'Income': True,
        'LoyaltyStatus': True,
        'Latitude': False,
        'Longitude': False
    },
    projection='mercator',
    color_continuous_scale=aiai_palette,
    title='Customer Lifetime Value Distribution — Amazing International Airlines (Canada)',
    scope='north america',
    size='Income',               # pontos maiores = maior rendimento
    size_max=15,
    opacity=0.75
)

# Configuração do mapa — foco e estilo
fig.update_geos(
    center=dict(lat=56.1304, lon=-106.3468),
    lataxis_range=[42, 72],
    lonaxis_range=[-140, -50],
    showland=True, landcolor="white",
    showcountries=True, countrycolor="lightgrey",
    showocean=True, oceancolor="#e8f1f8",
    lakecolor="#cfe5fa",
    showlakes=True,
    fitbounds="locations"
)

# Legenda (colorbar)
fig.update_layout(
    coloraxis_colorbar=dict(
        title=dict(
            text='Customer Lifetime Value',
            font=dict(size=14)
        ),
        tickfont=dict(size=12)
    )
)

fig.show()


# In[ ]:




