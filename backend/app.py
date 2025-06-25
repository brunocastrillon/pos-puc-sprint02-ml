from flask import Flask, json, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import os
import joblib
import pandas as pd

# -- carregando variaveis de ambiente
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

# -- Garante que ARTIFACT_PATH não seja vazio
if not MODEL_PATH:
    raise RuntimeError("A variável MODEL_PATH não está definida no .env")

# -- carregando o modelo
model = joblib.load(MODEL_PATH)

track_classification = model["track_classification"]
correlation_matrix = model["platform_correlation"]
artist_metrics   = model["artist_metrics"]
feature_names  = model["feature_names"]

app = Flask(__name__)
CORS(app)

@app.route("/predict-schema", methods=["GET"])
def predict_schema():
    return jsonify(feature_names)

@app.route("/predict-explicit", methods=["POST"])
def predict_explicit():
    payload = request.get_json(force=True)

    if not payload:
        return jsonify({"error":"JSON inválido"})
    
    dataframe = pd.DataFrame([payload])

    try:
        predict = track_classification.predict(dataframe)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    response = {"classified": bool(predict)}

    return jsonify(response)

@app.route("/platform-correlation", methods=["GET"])
def plataform_correlation():
    corr_df = correlation_matrix
    corr_json = {}

    for idx in corr_df.index:
        corr_json[str(idx)] = {}

        for col in corr_df.columns:
            val = corr_df.at[idx, col]

            if pd.isna(val):
                corr_json[str(idx)][str(col)] = None
            else:
                corr_json[str(idx)][str(col)] = float(val)
    
    return jsonify(corr_json)

@app.route("/artist-impact", methods=["GET"])
def get_artist_impact():
    records = artist_metrics.fillna(0).to_dict(orient="records")
    
    return jsonify(records)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)