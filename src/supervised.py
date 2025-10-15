import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    from data_ingestion import load_data, preprocess_data

    df = load_data("data/Advertising.csv")
    df = preprocess_data(df)

    variable_coloumn = df[["TV", "Radio", "Newspaper"]]
    target_column = df["Sales"]

    model = LinearRegression()
    model.fit(X_train, y_train)