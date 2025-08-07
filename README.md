# Churn Model with FastAPI

This project consists on creating a fastapi application, where this application will be responsible of predicting costumer retention. The response of the api will be made based on an Machine Learning model, that is pre-trained to understand some caracteristics from clients, that may show us if they will left the services or not.

## 📂 Project Structure
```
ml-api-fastapi/
├── app/
│   ├── main.py                 # API FastAPI
│   ├── models/
│   │   └── input.py            # Pydantic schema
│   ├── services/
│   │   └── predictor.py        # Lógica de carregamento e predição
│   ├── core/
│   │   ├── config.py
│   │   └── logger.py           # Logging rotativo
├── models/
│   └── model.pkl               # Modelo treinado
├── train/
│   └── train_model.py          # Script de treino e salvamento
├── logs/
│   └── app-2025-08-06.log      # (gerado automaticamente)
├── requirements.txt
└── README.md
```

## 🚀 Running locally

...

### 📋 Requirements

...

### 🔧 instalation

...

## ⚙️ Running Tests

...

## 📦 Deploy

...


## 📌 Version

...

## 📄 Licença

...