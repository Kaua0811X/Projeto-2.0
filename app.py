import json
import os
import subprocess

import joblib
from flask import Flask, jsonify, render_template, request

from utils.normalize import normalizar

app = Flask(__name__)


# --- Carregamento inicial do modelo e especialistas ---
def recarregar_modelo():
    global modelo_doenca
    global especialistas_dict

    modelo_doenca = joblib.load("modelo/modelo_doenca.pkl")  # Pipeline com vectorizer
    with open("modelo/especialistas.json", "r", encoding="utf-8") as f:
        especialistas_dict = json.load(f)


recarregar_modelo()


# --- Página principal ---
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    sintomas = ""

    if request.method == "POST":
        sintomas = request.form["sintomas"]
        correcao = request.form.get("correcao")

        sintomas_tratado = normalizar(sintomas)
        doenca_pt = modelo_doenca.predict([sintomas_tratado])[0]
        especialista = especialistas_dict.get(doenca_pt, "Clínico Geral")

        if correcao and correcao.strip():
            correcao = normalizar(correcao)
            novo_registro = {
                "sintomas": sintomas_tratado,
                "correcao": correcao
            }
            caminho_corrigido = "dados/correcoes_usuario.json"

            if os.path.exists(caminho_corrigido):
                with open(caminho_corrigido, "r", encoding="utf-8") as f:
                    dados_existentes = json.load(f)
            else:
                dados_existentes = []

            dados_existentes.append(novo_registro)

            with open(caminho_corrigido, "w", encoding="utf-8") as f:
                json.dump(dados_existentes, f, ensure_ascii=False, indent=2)

            doenca_pt = correcao
            especialista = especialistas_dict.get(doenca_pt, "Clínico Geral")

        resultado = {
            "sintomas": sintomas,
            "doenca_pt": doenca_pt,
            "especialista": especialista,
        }

    return render_template("index.html", resultado=resultado, **(resultado or {}))


# --- Re-treinamento do modelo ---
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        result = subprocess.run(
            ["python", "re_treinar_modelo.py"], capture_output=True, text=True
        )
        if result.returncode == 0:
            recarregar_modelo()
            return jsonify(
                {
                    "status": "ok",
                    "mensagem": "Modelo re-treinado e recarregado com sucesso!",
                }
            )
        else:
            return jsonify(
                {
                    "status": "erro",
                    "mensagem": "Erro ao treinar o modelo",
                    "detalhes": result.stderr,
                }
            ), 500
    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)}), 500


# --- Execução local ---
if __name__ == "__main__":
    app.run(debug=True)