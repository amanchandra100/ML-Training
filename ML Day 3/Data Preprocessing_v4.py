############# Data Pre-processing ##############

################ Type casting #################
import pandas as pd

data = pd.read_csv("ethnic diversity.csv")
data.dtypes

'''
EmpID is Integer - Python automatically identify the data types by interpreting the values. 
As the data for EmpID is numeric Python detects the values as int64.

From measurement levels prespective the EmpID is a Nominal data as it is an identity for each employee.

If we have to alter the data type which is defined by Python then we can use astype() function

'''

help(data.astype)

# Convert 'int64' to 'str' (string) type. 
data.EmpID = data.EmpID.astype('str')
data.dtypes

data.Zip = data.Zip.astype('str')
data.dtypes

# For practice:
# Convert data types of columns from:
    
# 'float64' into 'int64' type. 
data.Salaries = data.Salaries.astype('int64')
data.dtypes

# int to float
data.age = data.age.astype('float32')
data.dtypes


##############################################
### Identify duplicate records in the data ###
import pandas as pd
data = pd.read_csv("mtcars_dup.csv")

# Duplicates in rows
help(data.duplicated)

duplicate = data.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate

sum(duplicate)

# Parameters
duplicate = data.duplicated(keep = 'last')
duplicate

duplicate = data.duplicated(keep = False)
duplicate


# Removing Duplicates
data1 = data.drop_duplicates() # Returns DataFrame with duplicate rows removed.

# Parameters
data1 = data.drop_duplicates(keep = 'last')

data1 = data.drop_duplicates(keep = False)


# Duplicates in Columns
# We can use correlation coefficient values to identify columns which have duplicate information

import pandas as pd

cars = pd.read_csv("Cars.csv")

# Correlation coefficient
'''
Ranges from -1 to +1. 
Rule of thumb says |r| > 0.85 is a strong relation
'''
cars.corr() #corelation find karta hai

'''We can observe that the correlation value for HP and SP is 0.973 and VOL and WT is 0.999 
& hence we can ignore one of the variables in these pairs.
'''

################################################
############## Outlier Treatment ###############
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:/Users/mrper/OneDrive/Desktop/Machine_learning/Day2/ethnic diversity.csv")
df.dtypes

# Let's find outliers in Salaries
sns.boxplot(df.Salaries)

sns.boxplot(df.age)
# No outliers in age column

# Detection of outliers (find limits for salary based on IQR)
IQR = df['Salaries'].quantile(0.75) - df['Salaries'].quantile(0.25) # iqr = Q3-Q1

lower_limit = df['Salaries'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Salaries'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# Let's flag the outliers in the dataset

outliers_df = np.where(df.Salaries > upper_limit, True, np.where(df.Salaries < lower_limit, True, False))
df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# Let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Salaries)

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
df['df_replaced'] = pd.DataFrame(np.where(df['Salaries'] > upper_limit, upper_limit, np.where(df['Salaries'] < lower_limit, lower_limit, df['Salaries'])))
sns.boxplot(df.df_replaced)


############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr',tail = 'both',fold = 1.5,variables = ['Salaries']) 
                        # choose  IQR rule boundaries or gaussian for mean and std
                         # tail = 'both', # cap left, right or both tails 
                         # fold = 1.5,
                         # variables = ['Salaries'])

df_s = winsor_iqr.fit_transform(df[['Salaries']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(df_s.Salaries)


# Define the model with Gaussian method
winsor_gaussian = Winsorizer(capping_method = 'gaussian', tail = 'both', fold = 3,variables = ['Salaries']) 
                             # choose IQR rule boundaries or gaussian for mean and std
                          #tail = 'both', # cap left, right or both tails 
                         # fold = 3,
                         # variables = ['Salaries'])

df_t = winsor_gaussian.fit_transform(df[['Salaries']])
sns.boxplot(df_t.Salaries)


# Define the model with percentiles:
# Default values
# Right tail: 95th percentile
# Left tail: 5th percentile

winsor_percentile = Winsorizer(capping_method = 'quantiles',tail = 'both',fold = 0.05,variables = ['Salaries'])
                          #tail = 'both', # cap left, right or both tails 
                          #fold = 0.05, # limits will be the 5th and 95th percentiles
                          #variables = ['Salaries'])

df_p = winsor_percentile.fit_transform(df[['Salaries']])
sns.boxplot(df_p.Salaries)



