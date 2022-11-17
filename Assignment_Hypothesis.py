#!/usr/bin/env python
# coding: utf-8

# # Q1

# In[ ]:



#A F&B manager wants to determine whether there is any significant difference
#in the diameter of the cutlet between two units. A randomly selected sample of 
#cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level.
#Please state the assumptions and tests that you carried out to check validity of the assumptions.


# In[3]:


import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageGrab
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#We are going to conduct a 2 tailed t-Test on 2 Independent samples with Numerical Data
#We need to check whether the mean of both samples are different and
#Is there any significance difference between the two samples?


# In[5]:


#Step 1
#Make two Hypothesis one contradicting to other
#Null Hypothesis is want we want to prove
#Step 2
#Decide a cut-off value
#Significance 5%
#alpha = 0.05
#As it is a two-tailed test
#alpha/2 = 0.025


# In[9]:


#Step 3
#Collect evidence
#Importing Files
cutlets = pd.read_csv(r'C:\Users\anupa\Downloads\Cutlets.csv')
cutlets.head(10)


# In[10]:


#Applying Descriptive Statistics
cutlets.describe()


# In[11]:


#Checking for Null Values
cutlets.isnull().sum()


# In[12]:


cutlets[cutlets.duplicated()]


# In[13]:


#Checking the data type
cutlets.info()


# In[14]:


#Plotting the data
plt.subplots(figsize = (9,6))
plt.subplot(121)
plt.boxplot(cutlets['Unit A'])
plt.title('Unit A')
plt.subplot(122)
plt.boxplot(cutlets['Unit B'])
plt.title('Unit B')
plt.show()


# In[15]:


plt.subplots(figsize = (9,6))
plt.subplot(121)
plt.hist(cutlets['Unit A'], bins = 15)
plt.title('Unit A')
plt.subplot(122)
plt.hist(cutlets['Unit B'], bins = 15)
plt.title('Unit B')
plt.show()


# In[16]:


plt.figure(figsize = (8,6))
labels = ['Unit A', 'Unit B']
sns.distplot(cutlets['Unit A'], kde = True)
sns.distplot(cutlets['Unit B'],hist = True)
plt.legend(labels)


# In[17]:


#Plotting Q-Q plot to check whether the distribution follows normal distribution or not
sm.qqplot(cutlets["Unit A"], line = 'q')
plt.title('Unit A')
sm.qqplot(cutlets["Unit B"], line = 'q')
plt.title('Unit B')
plt.show()


# In[18]:


#Step 4
#Compare Evidences with Hypothesis using t-statistics
statistic , p_value = stats.ttest_ind(cutlets['Unit A'],cutlets['Unit B'], alternative = 'two-sided')
print('p_value=',p_value)


# In[19]:


#Compare p_value with ' '(Significane Level)
#If p_value is not_equal_to 'ALPHA ' we failed to reject Null Hypothesis because of lack of evidence
#If p_value is = 'ALPHA ' we reject Null Hypothesis
#interpreting p-value
alpha = 0.025
print('Significnace=%.3f, p=%.3f' % (alpha, p_value))
if p_value <= alpha:
    print('We reject Null Hypothesis there is a significance difference between two Units A and B')
else:
    print('We fail to reject Null hypothesis')


# In[ ]:


#Hence, We fail to reject Null Hypothesis because of lack of evidence, 
#there is no significant difference between the two samples


# In[20]:


ImageGrab.grabclipboard()


# # Q2

# In[4]:


#A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. 
#They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch.
#Step 3
#Collect evidence
#Importing Files
labtat = pd.read_csv(r'C:\Users\anupa\Downloads\LabTAT.csv')
labtat.head()#Analyze the data and determine whether there is any difference in average TAT among the different laboratories at 5% significance level.


# In[5]:


#Step 1
#Make two Hypothesis one contradicting to other
#Null Hypothesis is want we want to prove
#Step 2
#Decide a cut-off value
#Significance 5%
#alpha = 0.05


# In[6]:


#Step 3
#Collect evidence
#Importing Files
labtat = pd.read_csv(r'C:\Users\anupa\Downloads\LabTAT.csv')
labtat.head()


# In[7]:


#Applying Descriptive Statistics
labtat.describe()


# In[8]:


#Checking for Null Values
labtat.isnull().sum()


# In[9]:


#Checking for Duplicate Values
labtat[labtat.duplicated()].shape


# In[10]:


labtat[labtat.duplicated()]


# In[12]:


#Checking the data type
labtat.info()


# In[13]:


#Plotting the data
plt.subplots(figsize = (16,9))
plt.subplot(221)
plt.boxplot(labtat['Laboratory 1'])
plt.title('Laboratory 1')
plt.subplot(222)
plt.boxplot(labtat['Laboratory 2'])
plt.title('Laboratory 2')
plt.subplot(223)
plt.boxplot(labtat['Laboratory 3'])
plt.title('Laboratory 3')
plt.subplot(224)
plt.boxplot(labtat['Laboratory 4'])
plt.title('Laboratory 4')
plt.show()


# In[14]:


plt.subplots(figsize = (9,6))
plt.subplot(221)
plt.hist(labtat['Laboratory 1'])
plt.title('Laboratory 1')
plt.subplot(222)
plt.hist(labtat['Laboratory 2'])
plt.title('Laboratory 2')
plt.subplot(223)
plt.hist(labtat['Laboratory 3'])
plt.title('Laboratory 3')
plt.subplot(224)
plt.hist(labtat['Laboratory 4'])
plt.title('Laboratory 4')
plt.show()


# In[15]:


plt.figure(figsize = (8,6))
labels = ['Lab 1', 'Lab 2','Lab 3', 'Lab 4']
sns.distplot(labtat['Laboratory 1'], kde = True)
sns.distplot(labtat['Laboratory 2'],hist = True)
sns.distplot(labtat['Laboratory 3'],hist = True)
sns.distplot(labtat['Laboratory 4'],hist = True)
plt.legend(labels)


# In[16]:


#Plotting Q-Q plot to check whether the distribution follows normal distribution or not
sm.qqplot(labtat['Laboratory 1'], line = 'q')
plt.title('Laboratory 1')
sm.qqplot(labtat['Laboratory 2'], line = 'q')
plt.title('Laboratory 2')
sm.qqplot(labtat['Laboratory 3'], line = 'q')
plt.title('Laboratory 3')
sm.qqplot(labtat['Laboratory 4'], line = 'q')
plt.title('Laboratory 4')
plt.show()


# In[17]:


#Step 4
#eCompare Evidences with Hypothesis using t-statictic
test_statistic , p_value = stats.f_oneway(labtat.iloc[:,0],labtat.iloc[:,1],labtat.iloc[:,2],labtat.iloc[:,3])
print('p_value =',p_value)


# In[18]:


#Compare p_value with 'ALPHA '(Significane Level)
#If p_value is not_equal_to 'ALPHA ' we failed to reject Null Hypothesis because of lack of evidence
#If p_value is = 'ALPHA ' we reject Null Hypothesis
#interpreting p-value


# In[19]:


alpha = 0.05
print('Significnace=%.3f, p=%.3f' % (alpha, p_value))
if p_value <= alpha:
    print('We reject Null Hypothesis there is a significance difference between TAT of reports of the laboratories')
else:
    print('We fail to reject Null hypothesis')


# In[20]:


#Hence, We fail to reject Null Hypothesis because of lack evidence, there is no significant difference between the samples


# # Q3

# In[21]:


#Sales of products in four different regions is tabulated for males and females. 
#Find if male-female buyer rations are similar across regions


# In[ ]:


#Step 1
#Make two Hypothesis one contradicting to other
#Null Hypothesis is want we want to prove
#Null Hypothesis: There is no association or dependency between the gender based buyer rations across regions
#Alternative Hypthosis: There is a significant association or dependency between the gender based buyer rations across regions
#Step 2
#Decide a cut-off value
#Significance 5%
#alpha = 0.05
#As it is a one-tailed test
#alpha = 1-0.95 = 0.05


# In[23]:


#Step 3
#Collect evidence
#Importing Files
buyer = pd.read_csv(r'C:\Users\anupa\Downloads\BuyerRatio.csv', index_col = 0)
buyer


# In[24]:


table = [[50,142,131,70],
        [435,1523,1356,750]]


# In[26]:


#Applying Chi-Square X^2 contingency table to convert observed value into expected value
stat, p, dof, exp = stats.chi2_contingency(buyer) 
print(stat,"\n", p,"\n", dof,"\n", exp)


# In[27]:


stats.chi2_contingency(table) 


# In[28]:


observed = np.array([50, 142, 131, 70, 435, 1523, 1356, 750])
expected = np.array([42.76531299,  146.81287862,  131.11756787, 72.30424052, 442.23468701, 1518.18712138, 1355.88243213, 747.69575948])


# In[29]:


#Step 4
#Comparing Evidence with Hypothesis
statistics, p_value = stats.chisquare(observed, expected, ddof = 3)
print("Statistics = ",statistics,"\n",'P_Value = ', p_value)


# In[30]:


#Compare p_value with 'ALPHA '(Significane Level)
#If p_value is not_equal_to 'ALPHA ' we failed to reject Null Hypothesis because of lack of evidence
#If p_value is = 'ALPHA ' we reject Null Hypothesis
#interpreting p-value


# In[31]:


alpha = 0.05
print('Significnace=%.3f, p=%.3f' % (alpha, p_value))
if p_value <= alpha:
    print('We reject Null Hypothesis there is a significance difference between TAT of reports of the laboratories')
else:
    print('We fail to reject Null hypothesis')


# In[32]:


#We fail to reject Null Hypothesis because of lack evidence. Therefore, 
#there is no association or dependency between male-female buyers rations 
#and are similar across regions. Hence, Independent samples


# # Q4

# In[33]:


#TeleCall uses 4 centers around the globe to process customer order forms. 
#They audit a certain % of the customer order forms. Any error in order form renders it defective 
#and has to be reworked before processing. The manager wants to check whether the defective % varies by centre. 
#Please analyze the data at 5% significance level and help the manager draw appropriate inferences


# In[34]:


#We are going to conduct a Test of Independence using Chi-Square test with Contingency table
#We need to check whether the mean of any of these samples are different or the same?


# In[35]:


#Step 1
#Make two Hypothesis one contradicting to other
#Null Hypothesis is want we want to prove
#Step 2
#Decide a cut-off value
#Significance 5%
#alpha = 0.05


# In[40]:


#Step 3
#Collect evidence
#Importing Files
centers = pd.read_csv(r'C:\Users\anupa\Downloads\Costomer+OrderForm.csv')
centers.head(10)


# In[41]:


#Applying Descriptive Statistics
centers.describe()


# In[42]:


#Checking for Null Values
centers.isnull().sum()


# In[43]:


centers[centers.isnull().any(axis=1)]


# In[44]:


#Checking the data type
centers.info()


# In[49]:


#Checking value counts in data
print(centers['Phillippines'].value_counts(),'\n',centers['Indonesia'].value_counts(),'\n',centers['Malta'].value_counts(),'\n',centers['India'].value_counts())


# In[50]:


#Creating Contingency table
contingency_table = [[271,267,269,280],
                    [29,33,31,20]]
print(contingency_table)


# In[51]:


#Calculating Expected Values for Observed data
stat, p, df, exp = stats.chi2_contingency(contingency_table)
print("Statistics = ",stat,"\n",'P_Value = ', p,'\n', 'degree of freedom =', df,'\n', 'Expected Values = ', exp)


# In[52]:


#Defining Expected values and observed values
observed = np.array([271, 267, 269, 280, 29, 33, 31, 20])
expected = np.array([271.75, 271.75, 271.75, 271.75, 28.25, 28.25, 28.25, 28.25])


# In[53]:


#Step 4
#Compare Evidences with Hypothesis using t-statictic
test_statistic , p_value = stats.chisquare(observed, expected, ddof = df)
print("Test Statistic = ",test_statistic,'\n', 'p_value =',p_value)


# In[54]:


#Plotting the data
#Compare p_value with ' '(Significane Level)
#If p_value is not_equal_to 'ALPHA ' we failed to reject Null Hypothesis because of lack of evidence
#If p_value is = 'ALPHA' we reject Null Hypothesis
#interpreting p-value


# In[55]:


alpha = 0.05
print('Significnace=%.3f, p=%.3f' % (alpha, p_value))
if p_value <= alpha:
    print('We reject Null Hypothesis there is a significance difference between TAT of reports of the laboratories')
else:
    print('We fail to reject Null hypothesis')


# In[ ]:


#We fail to reject Null Hypothesis because of lack of evidence.

