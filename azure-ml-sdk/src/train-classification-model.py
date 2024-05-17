import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trainining-data", type=str)
    parser.add_argument("--reg_rate", type=float, default=0.01)
    
    args = parser.parse_args()
    return args

def main(args):
    

if __name__ == "__main__":
    args = parse_args()