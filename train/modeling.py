from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import optuna
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

from schemas import TrainingMetrics, TrainingResult
from settings import get_training_settings

MODEL_SEARCH_SPACES = {
    "catboost": lambda trial: {
        "iterations": trial.suggest_int("iterations", 300, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
        "loss_function": "Logloss",
        "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced"]),
        "logging_level": "Silent",
        "random_seed": 42,
    }
}

training_settings = get_training_settings()

def build_models(preprocessor=None):
    """Function to build linear and tree-based models.

    Args:
        preprocessor (ColumnTransformer): preprocessor for the model features.
    """
    if preprocessor is None:
        raise ValueError("At least one preprocessor must be provided.")
    
    return {
        "catboost": {
            "model": CatBoostClassifier,
            "preprocessor": preprocessor
        }
    }
    
def __optimize_model(model_name, models, x_train, y_train, x_val, y_val):
    
    def objective(trial):
        params = MODEL_SEARCH_SPACES[model_name](trial)
        
        model = models.get(model_name)["model"](**params)
        
        pipeline = Pipeline(
            steps=[
                ("preprocessor", models.get(model_name)["preprocessor"]),
                ("classifier", model)
            ]
        )

        pipeline.fit(x_train, y_train)
        
        preprocessor_fitted = pipeline.named_steps["preprocessor"]
        model_fitted = pipeline.named_steps["classifier"]
        
        x_val_transformed = preprocessor_fitted.transform(x_val)
        
        y_val_proba = model_fitted.predict_proba(x_val_transformed)[:, 1]
        
        return average_precision_score(y_val, y_val_proba)
            

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    
    return study.best_params, study.best_value

def __evaluate_pipeline(pipeline, X_test, y_test, threshold, best_value):
    """ Method for evaluating the pipeline on the test set and calculating the relevant metrics.

    Args:
        pipeline (_type_): The trained pipeline to evaluate
        X_test (_type_): Test features
        y_test (_type_): Test labels
        threshold (_type_): Classification threshold
        best_value (_type_): Best validation PR-AUC value

    Returns:
        _type_: Training metrics
    """
    optimized_pipeline_fitted = pipeline.named_steps["preprocess"]
    optimized_model_fitted = pipeline.named_steps["classifier"]
    
    x_test_transformed = optimized_pipeline_fitted.transform(X_test)
    
    y_test_proba = optimized_model_fitted.predict_proba(x_test_transformed)[:, 1]
    y_test_pred = (y_test_proba >= training_settings.threshold).astype(int)

    return TrainingMetrics(
        val_pr_auc_best=best_value,
        test_pr_auc=average_precision_score(y_test, y_test_proba),
        test_roc_auc=roc_auc_score(y_test, y_test_proba),
        test_precision=precision_score(y_test, y_test_pred),
        test_recall=recall_score(y_test, y_test_pred),
        test_f1_score=f1_score(y_test, y_test_pred),
        threshold=threshold,
    )
    
def train_and_log_models(models, X_train, y_train, X_test, y_test, X_val, y_val, X_trainval, y_trainval):
    """Function to train and log models using MLflow.

    Args:
        models (dict): dictionary of models and their preprocessors
        X_train (DataFrame): training features
        y_train (Series): training labels
        X_test (DataFrame): test features
        y_test (Series): test labels
        X_val (DataFrame): validation features
        y_val (Series): validation labels
        X_trainval (DataFrame): training + validation features
        y_trainval (Series): training + validation labels
    """
    for model_name in MODEL_SEARCH_SPACES.keys():
        print(f"Optimizing model: {model_name}")
        best_params, best_value = __optimize_model(model_name, models, X_train, y_train, X_val, y_val)
        print(f"Best params for {model_name}: {best_params}")
        print(f"Best Val PR-AUC for {model_name}: {best_value}\n")
        
        optimized_model = models.get(model_name)["model"](**best_params)
        optimized_pipeline = Pipeline(
            steps=[
                ("preprocess", models.get(model_name)["preprocessor"]),
                ("classifier", optimized_model)
            ]
        )
        
        with mlflow.start_run(run_name=f"{model_name} - Optimized"):
            
            optimized_pipeline.fit(X_trainval, y_trainval)
            
            metrics = __evaluate_pipeline(
                optimized_pipeline, 
                X_test, 
                y_test, 
                training_settings.threshold,
                best_value
            )
            
            mlflow.log_metrics(metrics.model_dump())
            
            mlflow.log_params(best_params)
            
            model_info = mlflow.sklearn.log_model(
                optimized_pipeline,
                name="model",
                registered_model_name=training_settings.mlflow_registered_model_name
            )
            
    return TrainingResult(
        model_name=model_name,
        model_uri=model_info.model_uri,
        metrics=metrics
    )