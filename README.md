# Churn Model with FastAPI

This project consists on creating a fastapi application, where this application will be responsible of predicting costumer retention. The response of the api will be made based on an Machine Learning model, that is pre-trained to understand some caracteristics from clients, that may show us if they will left the services or not. The data used for training is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. The API is built with FastAPI and exposes endpoints for health checks and churn prediction. The machine learning model is trained and saved using scripts in the `train/` directory.


## 📂 Project Structure
```
churn_model_fastapi/
├── app/
│   ├── main.py                 # FastAPI application entrypoint
│   ├── models/
│   │   └── input.py            # Pydantic schema for input validation
│   ├── routers/
│   │   ├── health_check.py     # Health check endpoint
│   │   └── prediction.py       # Prediction endpoint
│   ├── services/
│   │   └── predictor.py        # Model loading and prediction logic
│   └── core/
│       └── logger.py           # Rotating logger setup
├── models/
│   ├── churn_model.pkl         # Trained ML model
│   └── columns.pkl             # Model input columns
├── train/
│   └── train_model.py          # Model training and saving script
├── notebook/
│   └── EDA.ipynb               # Notebook created for exploratory data analysis, to understand about the data, relevant features and analyse what the best model for this project would be. 
├── logs/
│   ├── app.log                 # Application logs (rotated)
│   ├── access.log              # Gunicorn access logs
│   ├── error.log               # Gunicorn error logs
│   └── dont_remove             # Placeholder to keep logs folder in git
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker build instructions
├── gunicorn.conf.py            # Gunicorn server configuration
├── .env_example                # Example environment variables
└── README.md                   # Project documentation
```

## 🚀 What Does This Project Do?

- **API for Churn Prediction:** Receives customer data via a POST endpoint and returns the probability of churn.
- **Model Training:** The `train/train_model.py` script downloads the dataset from Kaggle, cleans it, trains a RandomForest model, and saves the model and its expected input columns.
- **Logging:** Rotating logs for both application and server events.
- **Dockerized:** Ready for containerized deployment with Gunicorn and Uvicorn workers.

## 🚀 Running locally

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd churn_model_fastapi
   ```

2. **Set up environment variables:**
   - Copy `.env_example` to `.env` and fill in your Kaggle credentials.

3. **Install dependencies:**
   ```sh
   uv pip install -r requirements.txt
   ```

4. **Train the model (if not already present):**
   ```sh
   python train/train_model.py
   ```

5. **Run the API locally:**
   ```sh
   gunicorn
   ```
   OR 
   ```sh
   python -m app.main
   ```
   - The API will be available at `http://localhost:8000`.

## 🐳 Running in Production with Docker

1. **Build the Docker image:**
   ```sh
   docker build -t churn-api .
   ```

2. **Run the container:**
   ```sh
   sudo docker run -p 8000:8000  -v $(pwd)/logs:/app/logs churn-api
   ```

   - The API will be available at `http://localhost:8000`.

   - The container uses Gunicorn with Uvicorn workers, as configured in [`gunicorn.conf.py`](gunicorn.conf.py).

   - The logs from container can be analyzed in "logs/" folder.

## 🛣️ API Endpoints

- `GET /health`  
  Returns service health status.

- `POST /predict`  
  Accepts a JSON payload matching the schema in [`InputData`](app/models/input.py) and returns the churn probability.
