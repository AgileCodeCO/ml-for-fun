# Import libraries
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def parse_args():
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", dest="train_data", type=str, required=True)
    parser.add_argument("--test-data", dest="test_data", type=str, required=True)
    args = parser.parse_args()
    
    return args

def main(args):
    print("Reading data...")
    train_df = read_data(args.train_data)
    test_df = read_data(args.test_data)
    
    categorical_columns = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
    numerical_columns = ["Age", "Annual_Premium"]
    
    print("Preprocessing data and applying encoding to categorical columns...")

    # Apply ordinal encoding to categorical columns
    for col in categorical_columns:
        encoder = OrdinalEncoder()
        train_df[col] = encoder.fit_transform(train_df[col].values.reshape(-1,1))
        test_df[col] = encoder.transform(test_df[col].values.reshape(-1,1))
    
    # Normalize the data
    for col in numerical_columns:
        scaler = MinMaxScaler()
        train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1,1))
        test_df[col] = scaler.transform(test_df[col].values.reshape(-1,1))
    
    X_train = train_df.drop('Response', axis=1)
    y_train = train_df['Response']
    
    # Split the data for validation
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.7, random_state=42)
    
    # Train using logistic regression
    model = train_logistic_regression_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Train using SVM
    model = train_svm_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
def train_logistic_regression_model(X_train, y_train):
    print("\n")
    print("Traning LogisticRegression model...")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm_model(X_train, y_train):
    print("Traning SVC model...")
    model = SVC(kernel='rbf', gamma='scale', C=1, max_iter=300, probability=true, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print(f"Evaluating {type(model).__class__.__name__} model...")
    
    predictions = model.predict(X_test)
    accuracy = np.average(y_test == predictions)
    print("Accuracy:", accuracy)
    
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:,1])
    print("AUC:", auc)
    print("*" * 30)

def read_data(data_path):
    # Read data
    data = pd.read_csv(data_path, index_col=0)
    return data

if __name__ == "__main__":
    
    print("\n")
    print("*" * 60)

    args = parse_args()
    main(args)
    
    print("*" * 60)
    print("\n")
