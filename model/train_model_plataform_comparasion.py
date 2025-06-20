import pandas as pd
import numpy as np
import joblib
import os

import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# === carrega variáveis de ambiente do arquivo .env
load_dotenv()

# === configurações de ambiente
SOURCE = os.getenv("DATA_SOURCE")
URL = os.getenv("DATA_URL")
FILE = os.getenv("DATA_FILE")
MODEL = os.getenv("MODEL_PATH")
TARGET = "Explicit Track"

# === Métricas por plataforma para comparação
PLATFORM_COLS = [
    "Spotify Streams", "Spotify Popularity",
    "YouTube Views", "YouTube Likes",
    "TikTok Views",   "TikTok Likes",
    "Pandora Streams","Pandora Track Stations",
    "Apple Music Playlist Count",
    "Deezer Playlist Count", "Amazon Playlist Count",
    "Soundcloud Streams", "Shazam Counts",
    "TIDAL Popularity", "SiriusXM Spins", "AirPlay Spins"
]

# === carrega os dados de acordo com o ambiente setado em config.env
def load_data() -> pd.DataFrame:
    if SOURCE == "url":
        print(f"Carregando dados de URL: {URL}")
        return pd.read_csv(URL, encoding='latin-1')
    elif SOURCE == "local":
        print(f"Carregando dados de arquivo local: {FILE}")
        return pd.read_csv(FILE, encoding='latin-1')
    else:
        raise ValueError("fonte de dados inválida")

# === remove vírgulas das colunas de plataforma e converte para float.
def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in PLATFORM_COLS:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("", np.nan)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# === converte a coluna alvo para int
def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe[TARGET] = dataframe[TARGET].astype(int)
    return dataframe

# === calcula a correlação entre as métricas de cada plataforma
def calculate_plataform_correlation(dataframe: pd.DataFrame) -> pd.DataFrame:
    colunas = [c for c in PLATFORM_COLS if c in dataframe.columns and not dataframe[c].isna().all()]
    return dataframe[colunas].corr()

def plot_plataform_correlation(dataframe: pd.DataFrame):
    plt.figure(figsize=(12, 10))
    sns.heatmap(dataframe, annot=True, cmap="viridis", fmt=".2f")
    plt.title("correlação entre as plataformas de streaming")
    plt.show()

def scatter_plataform_correlation(dataframe: pd.DataFrame, colunas: list) -> pd.DataFrame:
    sns.pairplot(dataframe[colunas].dropna(), kind="reg", plot_kws={"scatter_kws": {"s": 10}})
    plt.suptitle("pairplot de métricas por plataforma", y=1.02)
    plt.show()

# === retorna uma lista de colunas numéricas preditoras, excluíndo a coluna alvo
def get_feature_columns(dataframe: pd.DataFrame) -> list:
    colunas_numericas = dataframe.select_dtypes(include=[np.number]).columns.drop(TARGET)
    colunas_numericas_validas = [c for c in colunas_numericas if not dataframe[c].isna().all()]
    return colunas_numericas_validas

# === separando os conjutos de treino e teste
def split_data(dataframe: pd.DataFrame, features: list, target: str, test_size: float = 0.2, random_state: int = 42):
    x = dataframe[features]
    y = dataframe[target]

    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

# === define os pipelines para cada algoritimo e suas grades de hiperparâmetros
def build_pipelines():
    pipelines = {
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(class_weight="balanced", random_state=42))
        ]),
        "naive_bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", SVC(class_weight="balanced", probability=True, random_state=42))
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

# === realiza o grid-search com cross-validate para cada pipeline, retornando estimadores ajustados
def train_tune(pipelines: dict, params: dict, x_train: pd.DataFrame, y_train: pd.Series) -> dict:
    estimators = {}

    for name, pipeline in pipelines.items():
        print(f"treinamento {name}...")
        
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=params.get(name, {}),
            cv=5,
            scoring="accuracy", # scoring="f1_macro" / scoring="recall" / scoring="precision"
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(x_train, y_train)
        
        estimators[name] = grid.best_estimator_
        
        print(f"   -> melhor cv: {grid.best_score_:.4f}")
        print(f"   -> parametros: {grid.best_params_}")
    
    return estimators

# === avalia cada modelo, retorna a acurácia e imprime o relatórios
def evaluate(models: dict, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    results = {}

    for name, model in models.items():
        print(f"Avaliando {name}...")

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        print(f" Acurácia: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))
        print(confusion_matrix(y_test, y_pred))
        
        results[name] = acc

    return results

# === seleciona o melhor modelo baseado na acurácia
def select_best_model(results: dict, models: dict):
    best_name = max(results, key=results.get)
    best_model = models[best_name]
    best_score = results[best_name]

    print(f"Melhor modelo: {best_name} com acurácia de {best_score:.4f}")

    return best_model

# === salva o melhor modelo escolido
def save_model(model):
    joblib.dump(model, MODEL)
    print(f"Modelo salvo em: {MODEL}")

def main():
    # 1 - carrega e pré-processa o dataset
    dataframe = load_data()
    dataframe = clean_numeric_columns(dataframe)
    dataframe = preprocess(dataframe)

    # 2 - define as features e o split(dados de treinamento e de teste)
    features = get_feature_columns(dataframe)
    x_train, x_test, y_train, y_test = split_data(dataframe, features, TARGET)

    # 3 - monta o pipeline e parametros
    pipelines, params = build_pipelines()

    # 4 - realiza o treino e o ajuste
    models = train_tune(pipelines, params, x_train, y_train)

    # 5 - realiza a avaliação dos modelos treinados
    result = evaluate(models, x_test, y_test)

    # 6 - seleciona e salva o melhor modelo
    best_model = select_best_model(result, models)
    save_model(best_model)

    print("\n===\n")
    print("\nIniciando a comparação entre as plataformas\n")
    print("\n===\n")

    # 7 - comparação entre plataformas
    correlacao = calculate_plataform_correlation(dataframe)
    plot_plataform_correlation(correlacao)
    subset = ["Spotify Streams", "YouTube Views", "TikTok Views", "Pandora Streams"]
    scatter_plataform_correlation(dataframe, subset)    

if __name__ == "__main__":
    main()