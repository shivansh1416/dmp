#Loading the Wine and Iris datasets

import pandas as pd
from sklearn.datasets import load_wine, load_iris

# Load Wine dataset
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Load Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

print("Wine Dataset Head:")
print(wine_df.head())

print("\nIris Dataset Head:")
print(iris_df.head())

#Check if Attributes are Standardized

# Function to check if the data is standardized
def check_standardized(df):
    means = df.mean()
    stds = df.std()
    standardized = True
    
    for column in df.columns:
        if not (abs(means[column]) < 1e-5 and abs(stds[column] - 1) < 1e-5):
            standardized = False
            print(f"{column} is not standardized (mean: {means[column]:.2f}, std: {stds[column]:.2f})")
    
    if standardized:
        print("All attributes are standardized.")
    else:
        print("Some attributes are not standardized.")

# Check if Wine dataset is standardized
print("\nChecking if Wine dataset is standardized:")
check_standardized(wine_df)

# Check if Iris dataset is standardized
print("\nChecking if Iris dataset is standardized:")
check_standardized(iris_df)

#Standardize the Attributes if Necessary
from sklearn.preprocessing import StandardScaler

# Function to standardize the dataset
def standardize_dataset(df):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)
    standardized_df = pd.DataFrame(standardized_data, columns=df.columns)
    return standardized_df

# Standardize Wine dataset if necessary
print("\nStandardizing Wine dataset:")
wine_df_standardized = standardize_dataset(wine_df)
check_standardized(wine_df_standardized)

# Standardize Iris dataset if necessary
print("\nStandardizing Iris dataset:")
iris_df_standardized = standardize_dataset(iris_df)
check_standardized(iris_df_standardized)
