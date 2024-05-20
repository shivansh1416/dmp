#i

import pandas as pd

# Assuming dirty_iris dataset is already loaded into a pandas DataFrame called df
# For the sake of example, we'll create a small sample dataframe similar to dirty_iris

data = {
    'sepal_length': [5.1, 4.9, None, 4.6, 5.0, 'special_value', 7.0],
    'sepal_width': [3.5, 3.0, 3.2, 3.1, None, 3.6, 3.2],
    'petal_length': [1.4, 1.4, 1.3, 'special_value', 1.5, 1.0, 4.7],
    'petal_width': [0.2, 0.2, None, 0.2, 0.2, 0.2, 1.4],
    'species': ['setosa', 'setosa', 'versicolor', 'virginica', 'setosa', 'setosa', None]
}
df = pd.DataFrame(data)

# Calculate number and percentage of complete observations
complete_obs = df.dropna().shape[0]
total_obs = df.shape[0]
percentage_complete = (complete_obs / total_obs) * 100

print(f"Number of complete observations: {complete_obs}")
print(f"Percentage of complete observations: {percentage_complete:.2f}%")

#ii

# Replace special values with NA
df.replace('special_value', pd.NA, inplace=True)
print(df)

#iii

import json

# Read rules from file
with open('rules.txt', 'r') as file:
    rules = file.readlines()

# Print the resulting constraint object
print("Rules:")
for rule in rules:
    print(rule.strip())

#iv

# Define functions to check each rule
def check_species(species):
    return species in ['setosa', 'versicolor', 'virginica']

def check_positive(values):
    return all(value > 0 for value in values if pd.notna(value))

def check_petal_length_vs_width(petal_length, petal_width):
    if pd.isna(petal_length) or pd.isna(petal_width):
        return True
    return petal_length >= 2 * petal_width

def check_sepal_length(sepal_length):
    if pd.isna(sepal_length):
        return True
    return sepal_length <= 30

def check_sepal_vs_petal(sepal_length, sepal_width):
    if pd.isna(sepal_length) or pd.isna(sepal_width):
        return True
    return sepal_length > sepal_width

# Check each rule and count violations
violations = {
    'species': df['species'].apply(check_species).value_counts().get(False, 0),
    'positive_values': df.apply(lambda row: check_positive(row[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]), axis=1).value_counts().get(False, 0),
    'petal_length_vs_width': df.apply(lambda row: check_petal_length_vs_width(row['petal_length'], row['petal_width']), axis=1).value_counts().get(False, 0),
    'sepal_length': df['sepal_length'].apply(check_sepal_length).value_counts().get(False, 0),
    'sepal_vs_petal': df.apply(lambda row: check_sepal_vs_petal(row['sepal_length'], row['petal_length']), axis=1).value_counts().get(False, 0)
}

# Print and plot the result
import matplotlib.pyplot as plt

print("Rule Violations:")
for rule, count in violations.items():
    print(f"{rule}: {count} violations")

# Plot the result
plt.bar(violations.keys(), violations.values())
plt.title('Number of Violations per Rule')
plt.xlabel('Rule')
plt.ylabel('Number of Violations')
plt.show()

#v

import matplotlib.pyplot as plt
import numpy as np

# Convert sepal_length to numeric, coerce errors to NaN
df['sepal_length'] = pd.to_numeric(df['sepal_length'], errors='coerce')

# Plot boxplot
plt.boxplot(df['sepal_length'].dropna())
plt.title('Boxplot of Sepal Length')
plt.ylabel('Sepal Length')
plt.show()

# Find outliers using boxplot.stats
outliers = plt.boxplot(df['sepal_length'].dropna())['fliers'][0].get_ydata()
print(f"Outliers in sepal length: {outliers}")
