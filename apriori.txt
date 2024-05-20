
import pandas as pd
import warnings
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.frequent_patterns import apriori, association_rules

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load datasets
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target

# Binarize continuous data using KBinsDiscretizer
def binarize_data(df, n_bins=3):
    binarizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
    binarized_data = binarizer.fit_transform(df)
    binarized_df = pd.DataFrame(binarized_data, columns=df.columns).astype(int)
    # Convert to binary (0 and 1) by thresholding
    binarized_df = (binarized_df > 0).astype(int)
    return binarized_df

wine_binarized = binarize_data(wine_df)
iris_binarized = binarize_data(iris_df)

# Convert to one-hot encoded format for Apriori
wine_onehot = pd.get_dummies(wine_binarized)
iris_onehot = pd.get_dummies(iris_binarized)

# Function to run Apriori and generate rules
def run_apriori(data, min_support, min_confidence):
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules

# Run Apriori on Wine dataset
frequent_itemsets_wine_1, rules_wine_1 = run_apriori(wine_onehot, min_support=0.50, min_confidence=0.75)
frequent_itemsets_wine_2, rules_wine_2 = run_apriori(wine_onehot, min_support=0.60, min_confidence=0.60)

# Run Apriori on Iris dataset
frequent_itemsets_iris_1, rules_iris_1 = run_apriori(iris_onehot, min_support=0.50, min_confidence=0.75)
frequent_itemsets_iris_2, rules_iris_2 = run_apriori(iris_onehot, min_support=0.60, min_confidence=0.60)

# Evaluation function
def evaluate_rules(rules):
    print(f"Total Rules: {len(rules)}")
    if len(rules) > 0:
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        print("No rules found.")

# Evaluate and print results
print("Evaluation of Rules for Wine Dataset (min_support=0.50, min_confidence=0.75):")
evaluate_rules(rules_wine_1)

print("\nEvaluation of Rules for Wine Dataset (min_support=0.60, min_confidence=0.60):")
evaluate_rules(rules_wine_2)

print("\nEvaluation of Rules for Iris Dataset (min_support=0.50, min_confidence=0.75):")
evaluate_rules(rules_iris_1)

print("\nEvaluation of Rules for Iris Dataset (min_support=0.60, min_confidence=0.60):")
evaluate_rules(rules_iris_2)
