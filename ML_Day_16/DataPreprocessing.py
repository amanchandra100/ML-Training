# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:24:53 2024

@author: mrper
"""

import pandas as pd
import seaborn as sns

data = pd.read_csv("./credit.csv")
data.dtypes


duplicate = data.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate

pd.isnull(data).sum()

sum(duplicate)

sns.boxplot(data.months_loan_duration)

sns.boxplot(data.amount)

sns.boxplot(data.percent_of_income)
sns.boxplot(data.year_at_residence)
sns.boxplot(data.age)
sns.boxplot(data.existing_loans_count)
sns.boxplot(data.dependents)


#months_loan_duration
#amount
#age

# Detection of outliers (find limits for salary based on IQR)
#IQR = data['Salaries'].quantile(0.75) - df[''].quantile(0.25) # iqr = Q3-Q1
#IQR

# pip install sweetviz
import sweetviz # it use for Auto EDA
my_report = sweetviz.analyze([data,"data"])

my_report.show_html('Report.html')

