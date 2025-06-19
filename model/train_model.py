import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

# === setar os caminhos localhost ===
DIRETORIO_BASE = os.path.dirname(__file__)
CAMINHO_DATASET = os.path.join(DIRETORIO_BASE, "dataset", "most_streamed_spotify_songs_2024.csv")
CAMINHO_ARTEFATO = os.path.join(DIRETORIO_BASE, "artifacts")

os.makedirs(CAMINHO_ARTEFATO, exist_ok=True)

# === setar o caminho on-line ===
URL_DATASET = "https://raw.githubusercontent.com/brunocastrillon/pos-puc-sprint02-ml/master/model/dataset/most_streamed_spotify_songs_2024.csv"

# === carregar os dados ===
def carregar_dados(caminho_dataset):
    data_frame = pd.read_csv(caminho_dataset, sep=";", encoding='latin-1')

    print(f"Dimensões do dataset: {data_frame.shape}")
    data_frame.head(5)

    print(data_frame.dtypes)
    print(data_frame.isnull().sum())

    print(data_frame["Explicit Track"].value_counts(normalize=True))

    data_frame.describe().T

    correlacao = data_frame.select_dtypes(include="number").corr()
    correlacao["Explicit Track"].sort_values(ascending=False)

    data_frame["Explicit Track"] = data_frame["Explicit Track"].astype(int)    

    # === identificando colunas numericas
    colunas_numericas = (data_frame.select_dtypes(include=[np.number]).columns.drop("Explicit Track").tolist())

    # remove 'TIDAL Popularity' as it has all missing values
    if 'TIDAL Popularity' in colunas_numericas:
        colunas_numericas.remove('TIDAL Popularity')    

    print(f"Features numéricas selecionadas ({len(colunas_numericas)}):")
    print(colunas_numericas)

    # === separando rotulos e atributos
    x = data_frame[colunas_numericas]
    y = data_frame["Explicit Track"]

    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

def configurar_modelos():
    pass

def treinar_modelos():
    pass

def salvar_modelo():
    pass

if __name__ == "__main__":
    pass