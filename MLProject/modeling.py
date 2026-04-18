# import argparse
# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import joblib

# def train(data_path):
#     mlflow.start_run()

#     df = pd.read_csv(data_path)

#     X = df.drop("target", axis=1)
#     y = df["target"]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = LogisticRegression(max_iter=200)
#     model.fit(X_train, y_train)

#     preds = model.predict(X_test)
#     acc = accuracy_score(y_test, preds)

#     mlflow.log_metric("accuracy", acc)
#     mlflow.sklearn.log_model(model, "model")

#     joblib.dump(model, "trained_model.pkl")
#     mlflow.log_artifact("trained_model.pkl")

#     mlflow.end_run()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", type=str)
#     args = parser.parse_args()
#     train(args.data_path)

# ===

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

def train(data_path):
    mlflow.start_run()

    # Load dataset
    df = pd.read_csv(data_path)

    # Split features and labels
    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # Full Pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=200))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log results
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Save artifact
    joblib.dump(model, "trained_model.pkl")
    mlflow.log_artifact("trained_model.pkl")

    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    train(args.data_path)
