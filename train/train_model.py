import pandas as pd
import kagglehub
import glob
import numpy as np
import os
import logging
import joblib
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    log_loss
)

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the overall logging level

# Create a console handler and set its level and formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Only show INFO and above on console
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

load_dotenv()

class KaggleDatasetTraining():
    def __init__(self):
        """Initialize the KaggleDatasetTraining class."""
        
        mlflow.set_experiment("Telco Customer Churn Prediction")
        
        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")
        
        if not kaggle_username or not kaggle_key:
            raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment variables.")
        
    def fetch_kaggle_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Function to fetch a Kaggle dataset and transform it into a pandas DataFrame.

        Args:
            dataset_name (str): Name of the Kaggle dataset to be fetched

        Returns:
            pd.DataFrame: pandas DataFrame containing the data from the dataset
        """
        
        try:
            path = kagglehub.dataset_download(dataset_name)
            
            csv_files = glob.glob(f"{path}/*.csv")
            
            return pd.read_csv(csv_files[0])
        
        except ValueError as e:
            raise ValueError(f"Error fetching the dataset: {dataset_name}. Error: {str(e)}")
        
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {dataset_name}. Error: {str(e)}")
        
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function to clean the dataset, based on the EDA notebook.

        Args:
            df (DataFrame): pandas DataFrame containing the data from the dataset

        Returns:
            DataFrame: cleaned pandas DataFrame
        """
        
        # Example cleaning steps
        
        bin_cols = ["gender","Partner","Dependents","PhoneService","PaperlessBilling","Churn"]
        for col in bin_cols:
            df[col]=df[col].map({"Yes":1, "No":0, "Male":1, "Female":0})
            
        multi_cols= ["MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]

        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        logger.info(df.info())
        
        df = df.drop(['customerID'], axis=1)
        
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def train_model(self, df: pd.DataFrame):
        """Function to train the model and create an experiment tracking on MLFlow.

        Args:
            df (DataFrame): pandas DataFrame containing the data from the dataset
        Returns:
            classifier: trained model
            score: accuracy score of the model
        """
        
        x = df.drop(['Churn'], axis=1)
        y = df['Churn']
        
        x = pd.get_dummies(x, drop_first=True)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run(run_name="Telco_Churn_Classifier"):
            self.classifier = RandomForestClassifier()
            
            self.classifier.fit(x_train, y_train)
            
            y_pred_proba = self.classifier.predict_proba(x_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))
            mlflow.log_metric("pr_auc", average_precision_score(y_test, y_pred_proba))
            mlflow.log_metric("average_precision", average_precision_score(y_test, y_pred_proba))
            precision_, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            mlflow.log_metric("precision", precision_)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("log_loss", log_loss(y_test, y_pred_proba))

            mlflow.sklearn.log_model(self.classifier, "model")
        
        joblib.dump(x_train.columns.tolist(), f"{os.getcwd()}/models/columns.pkl")
        
    def optimize_hyperparameters(self, df: pd.DataFrame):
        """Function to optimize hyperparameters using Optuna.

        Args:
            x_train: training features
            y_train: training labels

        Returns:
            dict: best hyperparameters found
        """
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int('n_estimators', 50, 500),
                "max_depth": trial.suggest_int('max_depth', 3, 30),
                "min_samples_split": trial.suggest_int('min_samples_split', 2, 10),
                "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 5),
                "random_state": 42,
                "n_jobs": -1
            }

            model = RandomForestClassifier(**params)
            model.fit(x_train, y_train)
            
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            return roc_auc
        
        x = df.drop(['Churn'], axis=1)
        y = df['Churn']
        
        x = pd.get_dummies(x, drop_first=True)
        
        logger.info(x.info())
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        logger.info("Best Params:", study.best_params)
        logger.info("Best ROC_AUC:", str(study.best_value))
        
        best_params = study.best_params
        
        with mlflow.start_run(run_name="Best_Parameters_Run"):
            model = RandomForestClassifier(**best_params)
            model.fit(x_train, y_train)
            
            y_test_proba = model.predict_proba(x_test)[:, 1]
            
            mlflow.log_params(best_params)
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_test_proba))
            mlflow.log_metric("pr_auc", average_precision_score(y_test, y_test_proba))
            mlflow.log_metric("log_loss", log_loss(y_test, y_test_proba))
            
            mlflow.sklearn.log_model(model, "optimized_churn_model")
    
    def save_model(self, model, filename: str, path: str = "models/"):
        """Function to save the trained model to a local file.

        Args:
            model: trained model
            filename (str): name of the file where the model will be saved
        """
        try:
            joblib.dump(model, os.path.join(path, filename))
            
            
            logger.info(f"Model saved to {filename}")
            
        except Exception as e:
            raise Exception(f"Error saving the model: {filename}. Error: {str(e)}")


if __name__ == "__main__":    
    training = KaggleDatasetTraining()
    
    logger.info("======================================")
    logger.info("Getting dataset from Kaggle...")
    
    df = training.fetch_kaggle_dataset("blastchar/telco-customer-churn")
    
    logger.info(df.info())
    
    logger.info("Dataset fetched successfully.")
    
    logger.info("======================================")
    logger.info("Cleaning dataset...")
    
    df_cleaned = training.clean_dataset(df)
    
    logger.info("Preparing for training...")
    
    training.train_model(df_cleaned)
    
    training.optimize_hyperparameters(df_cleaned)
    
    logger.info("Training completed and model saved successfully.")
