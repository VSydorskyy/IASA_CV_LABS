from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2

from sklearn.pipeline import Pipeline

def df2array(
    df: pd.DataFrame,
    path_col: str,
    target_col: str
):
    X = []
    y = []

    for i in tqdm(range(df.shape[0])):
        image_name = df[path_col].iloc[i]
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        X.append(img.flatten())
        y.append(df[target_col].iloc[i])

    X = np.stack(X, axis=0)
    y = np.array(y)

    return X, y

def train_predict(
    sklearn_alg: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    path_col: str,
    target_col: str
):
    print("Train processing")
    X_train, y_train = df2array(train_df, path_col, target_col)
    print("Fit model")
    sklearn_alg.fit(X_train, y_train)
    print("Train predciting")
    y_train_pred = sklearn_alg.predict(X_train)
    print("Test processing")
    X_val, y_val = df2array(val_df, path_col, target_col)
    print("Test predciting")
    y_val_pred = sklearn_alg.predict(X_val)
    print("Ready!!!")
    return y_train_pred, y_val_pred, sklearn_alg