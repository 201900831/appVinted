#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# ML_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from cargar_datosFinal2 import load_and_prepare_data


# In[3]:


# Load and prepare data
def load_data(file_path):
    """
    Load and prepare the data from a CSV file.
    """
    df = load_and_prepare_data(file_path)
    return df

# Preprocess data
def preprocess_data(X):
    """
    Create a preprocessing pipeline to handle categorical features.
    """
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor

# Balance classes in the dataset
def balance_classes(df, target_column='will_buy'):
    """
    Balance classes by upsampling the minority class.
    """
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]

    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=len(df_majority), 
                                     random_state=42)
    
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df_balanced

# Train the model
def train_model(model, X_train, y_train):
    """
    Train a model with the given training data.
    """
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on the test set.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

# Main function to set up data, train and evaluate models
def run_models(file_path):
    """
    Load data, balance classes, preprocess, train, and evaluate models.
    Returns a DataFrame with evaluation results.
    """
    # Load and prepare data
    df = load_data(file_path)
    
    # Balance classes
    df_balanced = balance_classes(df, target_column="will_buy")

    # Define features and target
    X = df_balanced.drop(columns=["will_buy"], errors="ignore")
    y = df_balanced["will_buy"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    preprocessor = preprocess_data(X_train)

    # Define models
    models = {
        "Logistic Regression": Pipeline(steps=[('preprocessor', preprocessor), ('model', LogisticRegression(solver='liblinear', random_state=42))]),
        "Random Forest Classifier": Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
    }

    # Train and evaluate models
    results = []
    for model_name, pipeline in models.items():
        trained_model = train_model(pipeline, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)
        metrics["Model"] = model_name
        results.append(metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# In[4]:


# Example usage
if __name__ == "__main__":
    results_df = run_models("online_shoppers_intention.csv")  # Replace with your actual CSV file
    print("Resultados de evaluación de modelos:")
    print(results_df)

