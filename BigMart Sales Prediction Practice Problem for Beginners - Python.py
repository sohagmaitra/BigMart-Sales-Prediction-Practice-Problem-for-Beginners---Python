
# coding: utf-8

# In[3]:


#Loading Packages

import numpy as np                       #For mathematical calculations
import pandas as pd 
import seaborn as sns                    #For Data Visualization
import matplotlib.pyplot as plt          #For plotting graphs
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                          #To ignore any warnings
warnings.filterwarnings("ignore")


# In[4]:


#Reading data
train=pd.read_csv("C:/Users/Amitech/Documents/Data Science/Kaggle/BigMart Sales Prediction/Train.csv")
test=pd.read_csv("C:/Users/Amitech/Documents/Data Science/Kaggle/BigMart Sales Prediction/Test.csv")


# In[5]:


train.shape, test.shape


# In[9]:


train.columns


# In[7]:


test.columns


# In[10]:


#We need to predict Item_Outlet_Sales for given test data
#Let merge the train and test date for Exploratory Data Analysis

train['source']='train'
test['source']='test'
test['Item_Outlet_Sales']=0
data = pd.concat([train,test], sort =False)
print(train.shape, test.shape, data.shape)


# In[11]:


data['Item_Outlet_Sales'].describe()


# In[12]:


sns.distplot(data['Item_Outlet_Sales'])


# In[13]:


print('Skewness: %f' %data['Item_Outlet_Sales'].skew())
print('Kurtsis: %f' %data['Item_Outlet_Sales'].kurt())


# In[13]:


#lets look at numerical and categorical variables
data.dtypes


# In[15]:


categorical_features = data.select_dtypes(include=[np.object])
categorical_features.head(2)


# In[16]:


data['Outlet_Establishment_Year'].value_counts()


# In[17]:


#Find missing values
data.apply(lambda x: sum(x.isnull()))


# In[18]:


data.apply(lambda x: len(x.unique()))


# In[19]:


#frequency of categories
for col in categorical_features:
    print('\n%s column: '%col)
    print(data[col].value_counts())


# In[14]:


#Lets start looking Outlet_Size, Outlet_Location and Outlet_Type distribution in Item_Outlet_Sale

plt.figure(figsize = (10,9))

plt.subplot(311)
sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplot(312)
sns.boxplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplot(313)
sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 1.5)

plt.show()


# In[15]:


plt.figure(figsize = (14,9))

plt.subplot(211)
ax = sns.boxplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Outlet_Identifier vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplot(212)
ax = sns.boxplot(x='Item_Type', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Item_Type vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplots_adjust(hspace = 0.9, top = 0.9)
plt.setp(ax.get_xticklabels(), rotation=45)

plt.show()


# In[16]:


#Data Cleaning and Inputing Missing Values

item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

missing_values = data['Item_Weight'].isnull()
print('Missing values: %d' %sum(missing_values))

data.loc[missing_values, 'Item_Weight'] = data.loc[missing_values, 'Item_Identifier'].apply(lambda x: item_avg_weight.at
[x, 'Item_Weight'])
print('Missing values after immputation %d' %sum(data['Item_Weight'].isnull()))


# In[19]:


#Lets impute Outlet_Size with the mode of the Outlet_Size for the particular type of outlet

from scipy.stats import mode        #Import mode function

#Determining the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x:mode(x.astype('str')).
mode[0]))
print ('Mode for each Outlet_type:')
print (outlet_size_mode)

#Get a boolean variable specifying missing Item_Weight values
missing_values = data['Outlet_Size'].isnull()

#Impute data and check missing values and after imputation to confirm
print ('\nOrignal #missing: %d' %sum(missing_values))
data.loc[missing_values, 'Outlet_Size'] = data.loc[missing_values, 'Outlet_Type'].apply (lambda x: outlet_size_mode[x])
print (sum(data['Outlet_Size'].isnull()))


# In[21]:


#Modify Item_Visibility. 
#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Impute 0 values with mean visibility of the product:
missing_values = (data['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(missing_values))
data.loc[missing_values,'Item_Visibility'] = data.loc[missing_values,'Item_Identifier'].apply(lambda x: visibility_avg.at[x,
'Item_Visibility'])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))


# In[22]:


#Create a broad category of Type of Item

#get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[23]:


#Modify categories of Item_Fat_Content

#change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())

print('\.Modified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())


# In[27]:


plt.figure(figsize = (10,9))

plt.subplot(211)
sns.boxplot(x='Item_Type_Combined', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplot(212)
sns.boxplot(x='Item_Fat_Content', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4, top = 1.5)

plt.show()


# In[28]:


plt.subplot(211)
ax = sns.boxplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Outlet_Identifier vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplot(212)
ax = sns.boxplot(x='Item_Type', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Item_Type vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplots_adjust(hspace = 0.9, top = 0.9)
plt.setp(ax.get_xticklabels(), rotation=45)

plt.show()


# In[29]:


data.index = data['Outlet_Establishment_Year']
data.index


# In[32]:


df = data.loc[:,['Item_Outlet_Sales']]
df.head(2)


# In[33]:


data.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean().plot.bar()


# In[36]:


#Determine the years of operation of a store

data['Outlet_Years'] = 2009 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[37]:


data.index = data['Outlet_Establishment_Year']
df = data.loc[:,['Item_Outlet_Sales']]
ts = df['Item_Outlet_Sales']
plt.figure(figsize=(12,8))
plt.plot(ts, label='Item_Outlet_Sales')
plt.title('Outlet Establishment Year')
plt.xlabel('Time(year-month)')
plt.ylabel('Item_Outlet_Sales')
plt.legend(loc = 'best')
plt.show()


# In[38]:


plt.figure(figsize = (12,6))
ax = sns.boxplot(x = 'Outlet_Years', y = 'Item_Outlet_Sales', data = data)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
ax.set_title('Outlet years vs Item_Outlet_Sales')
ax.set_xlabel('', fontsize = 15)
ax.set_ylabel('Item_Outlet_Sales', fontsize = 15)

plt.show()


# In[39]:


temp_data = data.loc[data['Outlet_Establishment_Year'] ==1998]


# In[40]:


temp_data['Outlet_Type'].value_counts()


# In[41]:


test_temp_data = test.loc[test['Outlet_Establishment_Year'] == 1998]
test_temp_data['Outlet_Type'].value_counts()


# In[46]:


#numerical and One-Hot Coding of Categorical variables

#import library
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type', 'Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[47]:


#One Hot Coding
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined'])


# In[48]:


data.dtypes

