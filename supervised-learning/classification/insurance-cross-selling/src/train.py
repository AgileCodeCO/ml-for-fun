# Import libraries
import argparse
import mlflow
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
    
    mlflow.set_experiment("Insurance-CrossSell-Prediction")
    mlflow.sklearn.autolog(silent=True)
    
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
    regularization_rate = 0.1
    solver = 'liblinear'
    with mlflow.start_run(run_name=get_friendly_run_id("LogisticRegression_sag")):
        model = train_logistic_regression_model(X_train, y_train, regularization_rate, solver)
        evaluate_model(model, X_test, y_test)
    
    # Train using Random Forest
    n_estimators = 100
    criterion = 'gini'
    with mlflow.start_run(run_name=get_friendly_run_id("RandomForestClassifier")):
        model = train_random_forest_model(X_train, y_train, n_estimators, criterion)
        evaluate_model(model, X_test, y_test)
        
    
def train_logistic_regression_model(X_train, y_train, reg_rate, solver):
    print("Traning LogisticRegression model...")
    model = LogisticRegression(solver=solver, C=1/reg_rate, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(X_train, y_train, n_estimators, criterion):
    print("Traning RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, criterion=criterion)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print(f"Evaluating model...")
    
    predictions = model.predict(X_test)
    accuracy = np.average(y_test == predictions)
    print("Accuracy:", accuracy)
    mlflow.log_metric("evaluation_accuracy_score", accuracy)
    
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:,1])
    print("AUC:", auc)
    mlflow.log_metric("evaluation_roc_auc_score", auc)
    print("*" * 30)
    
def get_friendly_run_id(model_name = None):
    now = datetime.datetime.now()
    if(model_name):
        return now.strftime(f"%Y_%m_%d_%H_%M_%S-{model_name}")
    
    return now.strftime("%Y_%m_%d_%H_%M_%S")

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
