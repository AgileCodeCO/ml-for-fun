import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import mlflow

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--training-data", type=str)
    parser.add_argument("--reg_rate", type=float, default=0.01)
    
    args = parser.parse_args()
    return args

def get_data(path):
    print("Loading data from", path)
    df = pd.read_csv(path)
    
    return df

def split_data(df):
    print("Splitting data")
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, reg_rate):
    print("Training model")
    model = LogisticRegression(C=1/reg_rate, solver='liblinear')
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model")
    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    mlflow.log_metric("Accuracy", acc)

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    mlflow.log_metric("AUC", auc)

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    
    # Save the ROC curve artifact
    plt.savefig("ROC-Curve.png")
    mlflow.log_artifact("ROC-Curve.png")


def main(args):
    df = get_data(args.training_data)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    with mlflow.start_run():
        model = train_model(X_train, y_train, args.reg_rate)
        
        mlflow.log_param("regularization_rate", args.reg_rate)
        mlflow.sklearn.log_model(model, "LogisticRegression")
    
        evaluate_model(model, X_test, y_test)
    
    

if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)
    
    args = parse_args()
    main(args)

    print("*" * 60)
    print("\n\n")