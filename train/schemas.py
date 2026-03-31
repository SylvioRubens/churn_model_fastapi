from pydantic import BaseModel


class TrainingMetrics(BaseModel):
    val_pr_auc_best: float
    test_pr_auc: float
    test_roc_auc: float
    test_precision: float
    test_recall: float
    test_f1_score: float
    threshold: float


class TrainingResult(BaseModel):
    model_name: str
    model_uri: str
    registered_version: str | None = None
    metrics: TrainingMetrics