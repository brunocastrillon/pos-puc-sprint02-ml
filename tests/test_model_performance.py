import os
import pandas as pd
import numpy as np
import joblib
import pytest

from sklearn.metrics import accuracy_score, f1_score

# --- CONFIGURAÇÃO ---
DATA_FILE   = os.getenv(
    "DATA_FILE",
    "model/dataset/most_streamed_spotify_songs_2024_v2.csv"
)
MODEL_PATH  = os.getenv(
    "MODEL_PATH",
    "artifact_spotify_mvp_v2.pkl"
)
TARGET      = "Explicit Classified"

# Limiares de performance
MIN_ACCURACY    = 0.62   # 62%
MIN_WEIGHTED_F1 = 0.58   # 58%


@pytest.fixture(scope="module")
def test_data():
    if not os.path.exists(DATA_FILE):
        pytest.skip(f"Dataset não encontrado em {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, thousands=",", encoding="latin-1")
    df.dropna(axis=1, how="all", inplace=True)
    df[TARGET] = df[TARGET].astype(int)

    features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in [TARGET, "Explicit Track"]
    ]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[TARGET],
        test_size=0.2, stratify=df[TARGET], random_state=42
    )
    return X_test, y_test


def _load_model():
    assert os.path.exists(MODEL_PATH), f"Modelo não encontrado em {MODEL_PATH}"
    artifact = joblib.load(MODEL_PATH)
    # Se for pipeline puro
    from sklearn.base import BaseEstimator
    if hasattr(artifact, 'predict'):
        return artifact
    # Se for dict, busca o valor que implemente predict
    if isinstance(artifact, dict):
        for v in artifact.values():
            if hasattr(v, 'predict'):
                return v
    pytest.skip("Nenhum objeto de modelo com 'predict' encontrado no artefato")


def test_model_meets_accuracy(test_data):
    X_test, y_test = test_data
    model = _load_model()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= MIN_ACCURACY, (
        f"Acurácia abaixo do mínimo: {acc:.4f} (< {MIN_ACCURACY:.2f})"
    )


def test_model_meets_weighted_f1(test_data):
    X_test, y_test = test_data
    model = _load_model()
    y_pred = model.predict(X_test)
    f1w = f1_score(y_test, y_pred, average="weighted")
    assert f1w >= MIN_WEIGHTED_F1, (
        f"F1-score (ponderado) abaixo do mínimo: {f1w:.4f} (< {MIN_WEIGHTED_F1:.2f})"
    )
