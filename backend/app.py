from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

import os
import joblib
import pandas as pd

# -- carregando variaveis de ambiente
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

# -- carregando caminho do modelo
MODEL_PATH = os.getenv("MODEL_PATH", "output_model.pkl")

# -- carregando o modelo
model = joblib.load(MODEL_PATH)

track_classification = model["track_classification"]
correlation_matrix = model["platform_correlation"]
artist_metrics   = model["artist_metrics"]

app = Flask(__name__)

@app.route("/predict-explict", methods=["POST"])
def predict_explicit():
    payload = request.get_json()

    dataframe = pd.DataFrame([payload])
    predict = model.predict(dataframe)[0]

    return jsonify({"explict": bool(predict)})

@app.route("/plataform_correlation", methods=["GET"])
def plataform_correlation():
    return jsonify(plataform_correlation.to_dict())

@app.route("/artist_impact", methods=["GET"])
def get_artist_impact():
    return jsonify(artist_metrics.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)