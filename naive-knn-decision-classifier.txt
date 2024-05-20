import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load datasets
iris = load_iris()
wine = load_wine()

datasets = {'Iris': iris, 'Wine': wine}

# Function to evaluate classifiers
def evaluate_classifiers(X, y, test_size, scaling=False, cv=None):
    results = {}
    
    if scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    if cv:
        skf = StratifiedKFold(n_splits=cv)
        
    classifiers = {
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier()
    }
    
    for name, clf in classifiers.items():
        if cv:
            scores = cross_val_score(clf, X, y, cv=skf)
            results[name] = scores.mean()
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results[name] = accuracy_score(y_test, y_pred)
    
    return results

# Evaluation settings
settings = [
    {'test_size': 0.25, 'cv': None, 'scaling': False, 'description': '75% train, 25% test'},
    {'test_size': 0.333, 'cv': None, 'scaling': False, 'description': '66.6% train, 33.3% test'},
    {'test_size': 0.25, 'cv': None, 'scaling': True, 'description': '75% train, 25% test, scaled'},
    {'test_size': 0.333, 'cv': None, 'scaling': True, 'description': '66.6% train, 33.3% test, scaled'},
    {'test_size': None, 'cv': 10, 'scaling': False, 'description': '10-fold cross-validation'},
    {'test_size': None, 'cv': 10, 'scaling': True, 'description': '10-fold cross-validation, scaled'}
]

# Evaluate classifiers on both datasets
for dataset_name, dataset in datasets.items():
    print(f"Results for {dataset_name} dataset:")
    X, y = dataset.data, dataset.target
    for setting in settings:
        test_size = setting['test_size']
        cv = setting['cv']
        scaling = setting['scaling']
        description = setting['description']
        
        results = evaluate_classifiers(X, y, test_size, scaling, cv)
        print(f" - {description}:")
        for clf_name, accuracy in results.items():
            print(f"   {clf_name}: {accuracy:.4f}")
    print()
