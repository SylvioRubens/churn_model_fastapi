import pandas as pd
import kagglehub
import glob
import os
import logging
import mlflow
from sklearn.model_selection import train_test_split
from settings import get_training_settings as TrainingSettings
from logging_config import setup_train_logging


logger = setup_train_logging()
logger.setLevel(logging.DEBUG)  # Set the overall logging level

class KaggleDatasetTraining():
    def __init__(self):
        """Initialize the KaggleDatasetTraining class."""
        
        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")
        
        if not kaggle_username or not kaggle_key:
            raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment variables.")
        
    def __validate_schema(self, required_columns, df: pd.DataFrame) -> None:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        
    def __infer_feature_types(self, df: pd.DataFrame) -> None:
        """ set class variables for numeric and categorical features

        Args:
            df (pd.DataFrame): dataset to infer the feature types from
        """
        numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_features = [c for c in df.columns if c not in numeric_features + ["Churn"]]
        return numeric_features, categorical_features
        
    def fetch_kaggle_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Function to fetch a Kaggle dataset and transform it into a pandas DataFrame.

        Args:
            dataset_name (str): Name of the Kaggle dataset to be fetched

        Returns:
            pd.DataFrame: pandas DataFrame containing the data from the dataset
        """
        
        required_columns = {
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
        }
        
        try:
            path = kagglehub.dataset_download(dataset_name)
            
            csv_files = glob.glob(f"{path}/*.csv")
            
            data = pd.read_csv(csv_files[0])
            
            self.__validate_schema(required_columns, data)
            
            return data
        
        except ValueError as e:
            raise ValueError(f"Error fetching the dataset: {dataset_name}. Error: {str(e)}")
        
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {dataset_name}. Error: {str(e)}")
        
        
    def clean_and_transform_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method to clean the dataset, based on the EDA notebook.

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
        
        #Transforming boolean columns to integers
        boolean_cols = df.select_dtypes(include=['bool']).columns
        df[boolean_cols] = df[boolean_cols].astype(int)
        
        df = df.drop(['customerID'], axis=1)
        
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def split_dataset(self, df: pd.DataFrame, settings = TrainingSettings()) -> None:
        """Function to split the dataset into training, validation, and test sets. """
        
        X = df.drop("Churn", axis=1)
        y = df["Churn"]
        
        self.X_trainval, self.X_test, self.y_trainval, self.y_test = train_test_split(
            X,
            y,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=y,
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_trainval,
            self.y_trainval,
            test_size=settings.val_size,
            random_state=settings.random_state,
            stratify=self.y_trainval,
        )

        self.numeric_features, self.categorical_features = self.__infer_feature_types(df)