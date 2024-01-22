#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:27:52 2024

@author: alexandrcarr
"""

import pandas as pd

df = pd.read_excel('/Users/alexandrcarr/Desktop/DIss.xlsx')

df = df.iloc[:3509]


###################
## Data Cleaning ##
###################

# Creating Reference Categories For Dummy Variables
categories_to_drop = {
    'Ethnicity': 'White',
    'Gender': 'Male',
    'Min Education Requirements': 'High School',
    'Qualification': 'Low',
    'Income': 'Low',
    'Region': 'Prague'
}

# List of columns to create dummies for
columns_to_dummies = [
    'Qualification', 'Ethnicity', 'Job Category', 'Income',
    'Industry', 'Department', 'Min Education Requirements', 'Region',
    'Gender', 'Education'
]

# Dictionary to keep track of all the dropped dummies
all_dropped_dummies = {}

# Create dummy variables for each column in the list
for column in columns_to_dummies:
    # Create the dummy variables without dropping any
    dummies = pd.get_dummies(df[column], prefix=column.replace(' ', '_'), dtype=int)
    
    # Determine which category to drop
    if column in categories_to_drop:
        # If there's a specific category to drop, use it
        category_to_drop = categories_to_drop[column]
    else:
        # Otherwise, drop the first category
        category_to_drop = df[column].sort_values().unique()[0]
    
    # Record the dropped category
    all_dropped_dummies[column] = category_to_drop
    
    # Drop the specified category from the dummies
    category_col = f"{column.replace(' ', '_')}_{category_to_drop}"
    dummies.drop(columns=[category_col], inplace=True)
    
    # Drop the original column from the dataframe to avoid the name conflict when joining
    df = df.drop(columns=[column])
    
    # Rename the dummy columns to have the format 'OriginalName: DummyName'
    new_column_names = {original: f"{column}: {dummy.split('_')[-1]}" for original, dummy in zip(dummies.columns, dummies.columns)}
    dummies.rename(columns=new_column_names, inplace=True)
    
    # Join the dummy variables with the original dataframe
    df = df.join(dummies)

# Merging dummies into categories to ease analysis 

mergers = {
    "Food Production and Agriculture": ["Baking", "Meat", "Groceries", "Meats", "Food production and agriculture"],
    "Hotel and Restaurant Services": ["Hotels and restaurants", "Hotel", "Hotel ", "Restaraunts", "Restaurant"],
    "Heavy Industry Manufacturing": ["Heavy industry", "Heavy industry manufacturing", "Industry", "Inudstrials", "Steel  ", "Manufacturing", "Steel Trade"],
    "Retail Sales": ["Fashion", "Fast Food", "Retail sales"],
    "Luxury Retail Sales": ["Luxury retail sales"],
    "Healthcare and Pharmaceuticals": ["Pharma"],
    "Financial Services and Investment": ["Financial services"],
    "Energy Engineering and Maintenance": ["Energy"],
    "TMT": ["Technology"],
    "HR and Recruiting": ["Hr services"],
}

# Iterate over each merger and perform the merging
for merged_name, industries in mergers.items():
    # Create a column for the merged industry
    merged_col_name = f"Industry: {merged_name}"
    df[merged_col_name] = 0
    
    for industry in industries:
        original_col_name = f"Industry: {industry}"
        # Update the merged column if any of the industries in the merger has a 1
        df[merged_col_name] = df[merged_col_name] | df[original_col_name]
        # Drop the original column
        df.drop(columns=[original_col_name], inplace=True)

# Print out all the dropped dummies to verify
print("All dropped dummies:")
for column, category in all_dropped_dummies.items():
    print(f"{column}: {category}")
    
    

### Creating interaction terms: income and ethnicity 


# Interaction of "Income: High" with "Ethnicity: Roma"
df['Interaction_Income_High_Ethnicity_Roma'] = df['Income: High'] * df['Ethnicity: Roma']

# Interaction of "Income: High" with the implicit "Ethnicity: White" 
# (This will be the same as "Income: High" itself since "Ethnicity: White" is the reference category)
df['Interaction_Income_High_Ethnicity_White'] = df['Income: High']

# Interaction of "Income: Low" with "Ethnicity: Roma" 
# (This will be the same as "Ethnicity: Roma" itself since "Income: Low" is the reference category)
df['Interaction_Income_Low_Ethnicity_Roma'] = df['Ethnicity: Roma']



###############
## Modelling ##
###############

import statsmodels.api as sm
import os
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML


# Selecting the dependent variable
Y = df['Outcome']

# List of prefixes for your dummy variables
dummy_prefixes = [
    'Gender:',  'Ethnicity:', 'Region:', 'Job Category:', 'Income:', 'Industry:', 
    'Qualification:','Min_Education Requirements:'
]
# Selecting the independent variables (all the dummies for specified columns)
X = df[[col for col in df.columns if any(col.startswith(prefix) for prefix in dummy_prefixes)]]

# Ensure that independent variables are correctly identified
if X.empty:
    raise ValueError("No independent variables were found. Check your dummy variable prefixes.")

# Adding a constant to the model (for intercept)
X = sm.add_constant(X)

# Fit the OLS model
ols_model = sm.OLS(Y, X).fit()

# Fit the Logit model
logit_model = sm.Logit(Y, X).fit()


stargazer = Stargazer([ols_model, logit_model])

# Render the table as LaTeX
latex_output = stargazer.render_latex()

# Save the LaTeX code to a file
output_dir = '/Users/alexandrcarr/Desktop/'  # Update this to your desired directory
os.makedirs(output_dir, exist_ok=True)
latex_file_path = os.path.join(output_dir, 'model_summary.tex')

with open(latex_file_path, 'w') as f:
    f.write(latex_output)



#### alternative modelling [fighting multicollinearity with lasso regression]

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

# Selecting the dependent variable
Y = df['Outcome']

# List of prefixes for your dummy variables
dummy_prefixes = [
    'Gender', 'Ethnicity', 'Region', 'Job_Category', 'Income', 'Industry', 
    'Qualification','Min_Education_Requirements'
]

# Selecting the independent variables (all the dummies for specified columns)
X = df[[col for col in df.columns if any(prefix in col for prefix in dummy_prefixes)]]

# Ensure that independent variables are correctly identified
if X.empty:
    raise ValueError("No independent variables were found. Check your dummy variable prefixes.")

# Standardize the features before running LASSO since LASSO is sensitive to the scale of input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# Fit the LASSO model
lasso_model = Lasso(alpha=0.01)  # alpha is a hyperparameter and should be tuned using cross-validation
lasso_model.fit(X_scaled, Y)

# Fit the Logistic LASSO model
logistic_lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)  # C is inverse of regularization strength
logistic_lasso_model.fit(X_scaled, Y)

# Save the coefficients to a file
coefficients_file_path = os.path.join(output_dir, 'lasso_coefficients.txt')
with open(coefficients_file_path, 'w') as f:
    f.write("LASSO Model Coefficients:\n")
    for coef, col in zip(lasso_model.coef_, X.columns):
        f.write(f"{col}: {coef}\n")
    f.write("\nLogistic LASSO Model Coefficients:\n")
    for coef, col in zip(logistic_lasso_model.coef_[0], X.columns):
        f.write(f"{col}: {coef}\n")

coefficients_file_path









