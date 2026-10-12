"""
Entrenamiento del modelo de cancelaciones de tarjeta con MLflow.
Mónica María Ramírez Mejía (A01797493)

Uso:
    python train.py                 # valores por defecto (baseline del notebook)
    python train.py --C 0.1 --class-weight balanced
    python train.py --help
"""

import argparse
import os
import random

import kagglehub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT_NAME = "Credit Card Churn - Regresion Logistica"


def parse_args(argv=None):
    """Hiperparámetros y semilla como argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Entrena una Regresión Logística para predecir cancelación de clientes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverso de la regularización (menor = más regularización).")
    parser.add_argument("--penalty", choices=["l1", "l2"], default="l2",
                        help="Tipo de regularización.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="'balanced' compensa el desbalance de clases.")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Iteraciones máximas del optimizador.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Nombre de la corrida en MLflow.")
    return parser.parse_args(argv)


def set_seed(seed):
    """Fija la semilla para que los resultados sean reproducibles"""
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    """Descarga el dataset de Kaggle y lo lee"""
    path = kagglehub.dataset_download("sakshigoyal7/credit-card-customers")
    return pd.read_csv(os.path.join(path, "BankChurners.csv"))


def clean_data(df):
    """Limpieza definida en el notebook de exploración"""
    naive_cols = [c for c in df.columns if c.startswith("Naive_Bayes")]
    df = df.drop(columns=["CLIENTNUM", "Avg_Open_To_Buy"] + naive_cols)
    df["Attrition_Flag"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    return df


def split_data(df, seed):
    """One-hot encoding y partición 80/20 estratificada"""
    y = df["Attrition_Flag"]
    X = pd.get_dummies(df.drop(columns=["Attrition_Flag"]), drop_first=True)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def build_model(args):
    """Pipeline: escalado + Regresión Logística"""
    clf = LogisticRegression(
        C=args.C,
        penalty=args.penalty,
        solver="liblinear" if args.penalty == "l1" else "lbfgs",
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    return make_pipeline(StandardScaler(), clf)


def evaluate(y_test, y_pred):
    """Métricas de evaluación"""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def save_confusion_matrix(y_test, y_pred, path="confusion_matrix.png"):
    """Guarda la matriz de confusión como imagen"""
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Sigue", "Canceló"], cmap="Blues"
    )
    plt.title("Matriz de confusión")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df, args.seed)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("C", args.C)
        mlflow.log_param("penalty", args.penalty)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("seed", args.seed)

        model = build_model(args)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate(y_test, y_pred)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"{name}: {value:.3f}")

        mlflow.log_artifact(save_confusion_matrix(y_test, y_pred))
        mlflow.sklearn.log_model(model, name="modelo")


if __name__ == "__main__":
    main()
