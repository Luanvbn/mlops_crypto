import os
from dotenv import load_dotenv

load_dotenv()
print("Variáveis de ambiente carregadas com sucesso!")

import pandas as pd
import yfinance as yf
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- 1. Configuração do DagsHub e MLflow ---
# SUBSTITUA com seu usuário e nome de repositório!
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_API_KEY") or os.getenv("DAGSHUB_TOKEN")
REPO_NAME = os.getenv("DAGSHUB_REPO") or "mlops_crypto"
try:
    dagshub.init(DAGSHUB_USERNAME,
                 REPO_NAME,
                 url="https://dagshub.com/luanvbn/mlops_crypto.mlflow",
                 mlflow=True)
    print("DagsHub inicializado com sucesso!")
except Exception as e:
    print(f"Erro ao inicializar DagsHub: {e}")
    print("Verifique seu nome de usuário, nome do repositório e credenciais (DAGSHUB_USERNAME, DAGSHUB_API_KEY/DAGSHUB_TOKEN).")
    exit()

# --- 2. Preparação dos Dados (Série Temporal) ---

# Baixa 5 anos de dados do Bitcoin
ticker = "BTC-USD"
print(f"Baixando dados de {ticker}...")
data = yf.download(ticker, period="5y")

# 1. Selecionamos apenas o preço de fechamento
df = data[['Close']].copy()

# 2. Criamos nossa feature: 'Lag_1' (preço do dia anterior)
# O .shift(1) "empurra" a coluna 'Close' um dia para baixo
df['Lag_1'] = df['Close'].shift(1)

# 3. Remove a primeira linha, que agora tem um 'Lag_1' nulo (NaN)
df.dropna(inplace=True)

print("Dados processados com feature 'Lag_1':")
print(df.head())

# Define features (X) e target (y)
X = df[['Lag_1']] # Feature: Preço de ontem
y = df['Close']   # Target: Preço de hoje

# --- 3. Treinamento e Registro com MLflow ---

# O enunciado pede para iniciar o experimento.
print("Iniciando a execução do MLflow...")
with mlflow.start_run(run_name="Experimento_Crypto_Price") as run:
    
    # Parâmetros de divisão de dados
    test_size = 0.2
    shuffle_data = False # IMPORTANTE para séries temporais!
    
    # Divide os dados em treino e teste (SEM embaralhar)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=shuffle_data, # Garantindo a ordem cronológica
        random_state=42 # Apenas para reprodutibilidade
    )

    # Registra parâmetros (conforme o desafio)
    print("Registrando parâmetros...")
    mlflow.log_param("ticker", ticker)
    mlflow.log_param("features", "Lag_1")
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("shuffle", shuffle_data)

    # Treina o modelo
    print("Treinando o modelo...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Faz previsões
    y_preds = model.predict(X_test)

    # Calcula métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r2 = r2_score(y_test, y_preds)

    # Registra métricas (conforme o desafio)
    print("Registrando métricas...")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print(f"\n--- Resultados ---")
    print(f"  RMSE: {rmse:.2f} (Erro médio de previsão em USD)")
    print(f"  R2 Score: {r2:.4f}")
    print("--------------------")

    # (Opcional, mas recomendado) Salva o modelo
    #mlflow.sklearn.log_model(model, "model")
    
    print("Salvando modelo como artefato (workaround)...")
    
    # Vamos precisar da biblioteca joblib para salvar o modelo sklearn
    import joblib
    
    model_filename = "linear_regression_model.joblib"
    
    # 1. Salva o modelo no disco localmente
    joblib.dump(model, model_filename)
    
    # 2. Loga esse arquivo local como um artefato no MLflow
    # O 'artifact_path="model"' cria uma pasta "model" no DagsHub para organizar
    mlflow.log_artifact(model_filename, artifact_path="model") 
    
    # 3. (Opcional) Remove o arquivo local após o upload
    os.remove(model_filename)
    
    print("Modelo salvo com sucesso como artefato.")
    # --- FIM DO WORKAROUND ---

    print(f"\nExperimento concluído! Run ID: {run.info.run_id}")
    print("Verifique os resultados na aba 'Experiments' do seu repositório DagsHub.")