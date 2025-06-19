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

# === setar os caminhos ===
DIRETORIO_BASE = os.path.dirname(__file__)
CAMINHO_DATASET = os.path.join(DIRETORIO_BASE, "dataset", "most_streamed_spotify_songs_2024.csv")
CAMINHO_ARTEFATO = os.path.join(DIRETORIO_BASE, "artifacts")

os.makedirs(CAMINHO_ARTEFATO, exist_ok=True)

# === carregar os dados ===
def carregar_dados(caminho_dataset):
    data_frame = pd.read_csv(caminho_dataset, sep=";")

    # === identificando colunas numericas
    colunas_numericas = (data_frame.select_dtypes(include=[np.number]).columns.drop("Explicit Track").tolist())

    # === separando rotulos e atributos
    x = data_frame[colunas_numericas]
    y = data_frame["Explicit Track"]

