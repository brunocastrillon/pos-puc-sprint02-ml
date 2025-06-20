import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

# === carrega variáveis de ambiente do arquivo .env
load_dotenv()

# === configurações de ambiente
SOURCE = os.getenv("DATA_SOURCE")
URL = os.getenv("DATA_URL")
FILE = os.getenv("DATA_FILE")
MODEL = os.getenv("MODEL_PATH")
TARGET = "Explicit Track"

# === carrega os dados de acordo com o ambiente setado em config.env
def load_data() -> pd.DataFrame:
    if SOURCE == "url":
        print(f"Carregando dados de URL: {URL}")
        return pd.read_csv(URL, sep=";", encoding='latin-1')
    elif SOURCE == "local":
        print(f"Carregando dados de arquivo local: {FILE}")
        return pd.read_csv(FILE, sep=";", encoding='latin-1')
    else:
        raise ValueError("fonte de dados inválida")

# === converte a coluna alvo para int
def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe[TARGET] = dataframe[TARGET].astype(int)
    return dataframe

# === retorna uma lista de colunas numéricas preditoras, excluíndo a coluna alvo
def get_feature_columns(dataframe: pd.DataFrame) -> list:
    return dataframe.select_dtypes(include=[np.number]).columns.drop(TARGET).tolist()

# === separando os conjutos de treino e teste
def split_data(dataframe: pd.DataFrame, features: list, target: str, test_size: float = 0.2, random_state: int = 42):
    x = dataframe[features]
    y = dataframe[target]

    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

# === define os pipelines para cada algoritimo e suas grades de hiperparâmetros
def build_pipelines():
    pipelines = {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "naive_bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42))
        ]),                        
    }

    params = {
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9]
        },
        "decision_tree": {
            "model__max_depth": [3, 5, 10, None]
        },
        "naive_bayes":  {
            "model__var_smoothing": [1e-09, 1e-08, 1e-07]
        },
        "SVM":  {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        }      
    }

    return pipelines, params

