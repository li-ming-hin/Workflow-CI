import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train(data_path):
    mlflow.start_run()

    df = pd.read_csv(data_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "trained_model.pkl")
    mlflow.log_artifact("trained_model.pkl")

    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    train(args.data_path)
