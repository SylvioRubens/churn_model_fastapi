import os
import mlflow
import os
import argparse
from settings import get_training_settings
from data import KaggleDatasetTraining
from logging_config import setup_train_logging
from preprocessing import make_tree_preprocessor
from modeling import build_models, train_and_log_models
from registry import get_alias_version, get_latest_registered_model_version, promote_model_alias

mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
training_settings = get_training_settings()

def main(promote: bool = False):
    data = KaggleDatasetTraining()
    
    logger = setup_train_logging()
    
    logger.info("======================================")
    logger.info("Getting dataset from Kaggle...")
    
    df = data.fetch_kaggle_dataset("blastchar/telco-customer-churn")
    
    logger.info("Dataset fetched successfully.")
    
    logger.info("======================================")
    logger.info("Cleaning dataset...")
    
    cleaned_dataset = data.clean_and_transform_dataset(df)
    
    logger.info("Preparing for training...")
    
    data.split_dataset(cleaned_dataset)
    
    #  Creating a model dictionary containing the model and the preprocessor
    model_dict = build_models(preprocessor=make_tree_preprocessor(data.numeric_features, data.categorical_features))
    
    # training
    result = train_and_log_models(
        model_dict, 
        data.X_train, 
        data.y_train, 
        data.X_test, 
        data.y_test, 
        data.X_val, 
        data.y_val, 
        data.X_trainval, 
        data.y_trainval
    )
    
    logger.info("=======================================")
    logger.info(f"Training completed for model: {result.model_name}")
    logger.info(f"logged model: {result.model_name}, test_pr_auc: {result.metrics.test_pr_auc:.4f}, test_recall: {result.metrics.test_recall:.4f}")
    
    
    if promote:
        version_to_promote = get_latest_registered_model_version(training_settings.mlflow_registered_model_name)
        promote_model_alias(
            training_settings.mlflow_registered_model_name, 
            version_to_promote, 
            "champion")
        logger.info(f"Promoted Model: {result.model_name}, version: {version_to_promote} to `champion`")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Whether to automatically promote the trained model to champion after training"
    )
    
    args = parser.parse_args()
    main(promote=args.promote)