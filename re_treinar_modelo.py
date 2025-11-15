import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from utils.normalize import normalizar

# Caminhos
JSON_BASE = "dados/doencas_sintomas_especialistas_pt.json"
JSON_CORRECOES = "dados/correcoes_usuario.json"
PASTA_MODELO = "modelo"

# Carrega JSON base
with open(JSON_BASE, "r", encoding="utf-8") as f:
    dados_base = json.load(f)

linhas = []
especialistas_dict = {}

# Processa dados base
for item in dados_base:
    doenca = item["doenca"]
    especialista = item.get("especialista", "Clínico Geral")
    especialistas_dict[normalizar(doenca)] = normalizar(especialista)

    for sintoma in item["sintomas"]:
        linhas.append({"symptom": sintoma.lower().strip(), "diagnosis": doenca})

# Adiciona correções do usuário (se houver)
if os.path.exists(JSON_CORRECOES):
    with open(JSON_CORRECOES, "r", encoding="utf-8") as f:
        correcoes = json.load(f)

    # Se for dict único, transforma em lista
    if isinstance(correcoes, dict):
        correcoes = [correcoes]

    df_corr = pd.DataFrame(correcoes)

    if "sintomas" in df_corr.columns and "correcao" in df_corr.columns:
        df_corr = df_corr[["sintomas", "correcao"]]
        df_corr.columns = ["symptom", "diagnosis"]
        df_corr["symptom"] = df_corr["symptom"].apply(normalizar)
        df_corr["diagnosis"] = df_corr["diagnosis"].apply(normalizar)

        linhas.extend(df_corr.to_dict("records"))
        print(f"✅ Correções aplicadas: {len(df_corr)} registros")
    else:
        print("⚠️ Arquivo de correções não contém os campos esperados.")

# Cria DataFrame final
df = pd.DataFrame(linhas)
print(f"🔢 Total de exemplos para treino: {len(df)}")

# Treinamento
X = df["symptom"]
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline(
    [("vectorizer", TfidfVectorizer()), ("classifier", MultinomialNB())]
)

pipeline.fit(X_train, y_train)

# Avaliação
y_pred = pipeline.predict(X_test)
print("📊 Acurácia:", accuracy_score(y_test, y_pred))
print("📋 Relatório:\n", classification_report(y_test, y_pred))

# Salva arquivos
os.makedirs(PASTA_MODELO, exist_ok=True)
joblib.dump(pipeline, os.path.join(PASTA_MODELO, "modelo_doenca.pkl"))
joblib.dump(pipeline.named_steps["vectorizer"], os.path.join(PASTA_MODELO, "vetor.pkl"))
with open(os.path.join(PASTA_MODELO, "especialistas.json"), "w", encoding="utf-8") as f:
    json.dump(especialistas_dict, f, ensure_ascii=False, indent=2)

print("✅ Modelo re-treinado e salvo com sucesso.")
