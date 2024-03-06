#!/usr/bin/env python
# coding: utf-8

# # Bank Customer Churn Prediction

# ### Dataset
# The dataset comprises 10,000 records and 14 variables, aimed at determining the likelihood of a bank's customers discontinuing their services. It includes both demographic and financial attributes of customers to facilitate this prediction.
# 
# 
# ### Data Dictionary
# | Column Name | Description |
# | --- | --- |
# | RowNumber | Row number |
# | CustomerId | Unique identification key for different customers |
# | Surname | Customer's last name |
# | CreditScore | Credit score of the customer |
# |Geography | Country of the customer |
# |Age | Age of the customer |
# |Tenure | Number of years for which the customer has been with the bank |
# |Balance | Bank balance of the customer |
# |NumOfProducts | Number of bank products the customer is utilising |
# |HasCrCard | Binary flag for whether the customer holds a credit card with the bank or not |
# |IsActiveMember | Binary flag for whether the customer is an active member with the bank or not |
# |EstimatedSalary | Estimated salary of the customer in Dollars |
# |Exited | Binary flag 1 if the customer closed account with bank and 0 if the customer is retained |

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading the dataset
df = pd.read_csv('churn.csv')
df.head()


# ### Data Preprocessing 

# In[4]:


#checking the shape of the dataset
df.shape


# ### Dropping the unecessary columns - RowNumber, CustomerId, Surname

# In[5]:


#drop coulumns
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)


# ### Checking for Null or Missing values

# In[6]:


#null values count
df.isnull().sum()


# ### Datatypes

# In[7]:


#column data types
df.dtypes


# ### Duplicate values

# In[8]:


#dulicate values
df.duplicated().sum()


# ### Renaming the column 

# In[9]:


#rename column
df.rename(columns={'Exited':'Churn'}, inplace=True)


# ### Descriptive Statistics

# In[9]:


#descriptive statistics
df.describe()


# In[10]:


df.head()


# ### Explorative Data Analysis

# During the exploratory data analysis phase, I will examine the dataset's distribution, investigate how the features correlate with the target variable, and explore the interactions between the features and the target variable. The analysis will begin with an assessment of the data distribution, then proceed to analyze the connections between the features and the target variable.

# In[11]:


#pie chart
plt.figure(figsize=(10,6))
plt.pie(df['Churn'].value_counts(),labels=['No','Yes'],autopct='%1.2f%%')
plt.title('Churn Percentage')
plt.show()


# The pie chart effectively illustrates the customer churn present in the dataset, showing that a significant majority of the customers remain with the bank, while only 20.4% have decided to leave.

# In[12]:


#gender and customer churn
sns.countplot(x = 'Gender', data = df, hue = 'Churn')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# The graph demonstrates that most customers are male. However, an analysis of customer churn reveals that females are slightly more inclined to leave the bank than males. Nonetheless, the difference in churn rates between the genders is minimal, indicating that gender alone may not be a reliable predictor of customer churn.

# In[13]:


#histogram for age distribution
sns.histplot(data=df, x="Age", hue="Churn", multiple="stack",kde=True)


# The histogram showcases the distribution of customer ages alongside their churn numbers. The largest segment of customers is aged between 30 and 40 years. However, it's the 40 to 50 age group that exhibits the highest frequency of churn. On the other hand, the 20 to 25 age group has the lowest incidence of churn. Thus, age emerges as a critical factor in predicting customer churn, indicating that those in later adulthood are more prone to discontinuing their banking services, unlike younger adults who have the least likelihood of churning.

# #### Credit Score

# In[14]:


fig, ax = plt.subplots(1,2,figsize=(15, 5))
sns.boxplot(x="Churn", y="CreditScore", data=df, ax=ax[0])
sns.violinplot(x="Churn", y="CreditScore", data=df, ax=ax[1])


# The boxplot and violin plot illustrate the credit score distribution of customers in relation to their churn status. In the boxplot, the medians for both churned and retained customers are nearly identical. Similarly, the violin plot's shape does not significantly differ between customers who have churned and those who have not. Although a subset of customers who churned exhibit lower credit scores, the overall analysis suggests that credit score alone does not serve as a reliable predictor of customer churn.

# #### Customer location

# In[15]:


sns.countplot(x = 'Geography', hue = 'Churn', data = df)
plt.title('Geography and Churn')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()


# The graphs display the distribution of customers by their respective countries alongside the churn rates. France accounts for the majority of the customer base, with Spain and Germany following. In contrast, the churn rate is highest in Germany, then France, and lowest in Spain. This suggests that customers from Germany are more prone to discontinuing their services compared to those from other countries.

# #### Tenure

# In[16]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='Tenure', data=df,ax=ax[0])
sns.countplot(x='Tenure', hue='Churn', data=df,ax=ax[1])


# Tenure, defined as the duration (in years) a customer has been with the bank, shows that most customers in the dataset fall within the 1-9 years range, distributed relatively evenly across these years. There are notably fewer customers with a tenure of less than 1 year or more than 9 years. Analyzing customer churn by tenure reveals that those within the 1-9 year bracket exhibit higher churn rates, peaking with customers at 1 year of tenure and similarly high for those at 9 years. Conversely, customers with a tenure exceeding 9 years demonstrate the lowest churn rates, suggesting greater loyalty to the bank and a decreased likelihood of churning.

# #### Bank Balance

# In[17]:


sns.histplot(data=df, x="Balance", hue="Churn", multiple="stack",kde=True)


# A significant portion of customers with a bank balance of zero also tend to discontinue their banking services. Moreover, those with a bank balance ranging from 100,000 to 150,000 are the next most likely group to leave the bank, following the customers with zero balance.

# #### Number of products purchased

# In[18]:


sns.countplot(x='NumOfProducts', hue='Churn', data=df)


# The dataset categorizes customers based on the number of products they have purchased, dividing them into four groups. The majority of customers fall into the groups that have bought either one or two products, and these groups show a lower churn rate compared to their non-churning counterparts. Conversely, for the categories of customers who have acquired three or four products, the churn rate is significantly higher than that of customers who remain with the bank. This indicates that the number of products a customer purchases serves as a reliable predictor of potential churn.

# #### Customers with or without credit card

# In[19]:


sns.countplot(x=df['HasCrCard'],hue=df['Churn'])


# The majority of the customers, approximately 70%, possess credit cards, leaving about 30% without. Additionally, a higher proportion of customers who decide to leave the bank are those who own a credit card.

# #### Active Members

# In[20]:


sns.countplot(x='IsActiveMember', hue='Churn', data=df)


# As anticipated, inactive bank members exhibit a higher churn rate compared to active members. This trend likely reflects greater satisfaction with the bank's services among active members, reducing their propensity to churn. Consequently, the bank should prioritize engagement with inactive members and enhance its services to encourage their retention.

# #### Estimated Salary

# In[21]:


sns.histplot(data=df,x='EstimatedSalary',hue='Churn',multiple='stack',palette='Set2')


# The graph depicts the distribution of customers' estimated salaries alongside the churn statistics. Overall, there is no discernible pattern distinguishing the salary distributions between customers who have churned and those who have not. Thus, estimated salary does not serve as a reliable indicator for predicting customer churn.

# #### Label encoding the variables

# In[22]:


#label encoding
variables = ['Geography','Gender']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in variables:
    le.fit(df[i].unique())
    df[i]=le.transform(df[i])
    print(i,df[i].unique())


# #### Normalization

# In[46]:


#normalize the continuous variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['CreditScore','Balance','EstimatedSalary']] = scaler.fit_transform(df[['CreditScore','Balance','EstimatedSalary']])


# In[47]:


df.head()


# ### Coorelation Matrix Heatmap

# In[48]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# There is no significant coorelation among the variables. So, I will proceed to model building.

# ### Train Test Split

# In[49]:


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop('Churn',axis=1),df['Churn'],test_size=0.3,random_state=42)


# ### Churn Prediction
# For predicting the churn of customers, depending on the data of the customers, we will use the following models:
# - Decision Tree Classifier
# - Random Forest Classifier

# ### Decision Tree Classifier

# Using GridSearchCV to find the best parameters for the model.

# In[50]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
#creating Decision Tree Classifer object
dtree = DecisionTreeClassifier()

#defining parameter range
param_grid = {
    'max_depth': [2,4,6,8,10,12,14,16,18,20],
    'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]
    }

#Creating grid search object
grid_dtree = GridSearchCV(dtree, param_grid, cv = 5, scoring = 'roc_auc', n_jobs = -1, verbose = 1)

#Fitting the grid search object to the training data
grid_dtree.fit(X_train, y_train)

#Printing the best parameters
print('Best parameters found: ', grid_dtree.best_params_)


# Adding the parameters to the model

# In[104]:


dtree = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42, min_samples_leaf=10)
dtree


# In[105]:


#training the model
dtree.fit(X_train,y_train)
#training accuracy
dtree.score(X_train,y_train)


# Predicting Customer Churn from Test set

# In[106]:


dtree_pred = dtree.predict(X_test)


# ### Random Forest Classifier

# In[127]:


from sklearn.ensemble import RandomForestClassifier
#creating Random Forest Classifer object
rfc = RandomForestClassifier()

#defining parameter range
param_grid = {
    'max_depth': [2,4,6,8,10],
    'min_samples_leaf': [2,4,6,8,10],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]
    }

#Creating grid search object
grid_rfc = GridSearchCV(rfc, param_grid, cv = 5, scoring = 'roc_auc', n_jobs = -1, verbose = 1)

#Fitting the grid search object to the training data
grid_rfc.fit(X_train, y_train)

#Printing the best parameters
print('Best parameters found: ', grid_rfc.best_params_)


# Adding the parameters to the model

# In[129]:


rfc = RandomForestClassifier(min_samples_leaf=8, max_depth=10, random_state=0, criterion='entropy')
rfc


# In[130]:


#training the model
rfc.fit(X_train, y_train)
#model accuracy
rfc.score(X_train, y_train)


# Predicting the customer churn from Test set

# In[131]:


rfc_pred = rfc.predict(X_test)


# ### Model Evalution

# ### Decision Tree Classifier

# #### Confusion Matrix Heatmap

# In[107]:


#confusion matrix heatmap
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test,dtree_pred),annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Decision Tree')
plt.show()


# The True Positive shows the count of correctly classified data points whereas the False Positive elements are those that are misclassified by the model. The higher the True Positive values of the confusion matrix the better, indicating many correct predictions.

# #### Distribution Plot

# In[108]:


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(dtree_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


# The more overlapping of two colors, the more accurate the model is.

# #### Classification Report

# In[109]:


from sklearn.metrics import classification_report
print(classification_report(y_test, dtree_pred))


# In[110]:


from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
print("Accuracy Score: ", accuracy_score(y_test, dtree_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, dtree_pred))
print("R2 Score: ", r2_score(y_test, dtree_pred))


# ### Random Forest Classifier

# #### Confusion Matrix Heatmap

# In[132]:


plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test,rfc_pred),annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()


# The True Positive shows the count of correctly classified data points whereas the False Positive elements are those that are misclassified by the model. The higher the True Positive values of the confusion matrix the better, indicating many correct predictions.

# #### Distribution Plot

# In[133]:


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(rfc_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


# #### Classification Report

# In[134]:


from sklearn.metrics import classification_report
print(classification_report(y_test, rfc_pred))


# In[135]:


print("Accuracy Score: ", accuracy_score(y_test, rfc_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, rfc_pred))
print("R2 Score: ", r2_score(y_test, rfc_pred))


# ## Conclusion
# The exploratory data analysis has led to the conclusion that customer churn is influenced by the following factors:
# 1. Age
# 2. Geography
# 3. Tenure
# 4. Balance
# 5. Number of Products
# 6. Has Credit Card
# 7. Is Active Member
# 
# Regarding the classification models, the following models were utilized:
# 1. Decision Tree Classifier
# 2. Random Forest Classifier
# 
# Hyperparameter tuning for both models was conducted using GridSearchCV. While their accuracy scores are comparable, the Random Forest Classifier outperforms the Decision Tree Classifier in terms of both accuracy and precision.
