
import mlflow

# Connect to remote MLflow server
mlflow.set_tracking_uri(
    "https://dagshub.com/Soummojeet-07/Ml_Flow_Experiment.mlflow"
)

# Select experiment
mlflow.set_experiment("Wine Quality Prediction")

# Start tracking
with mlflow.start_run():

    mlflow.log_param("alpha", 0.5)
    mlflow.log_param("l1_ratio", 0.5)

    mlflow.log_metric("rmse", 0.79)
    mlflow.log_metric("r2_score", 0.85)

    print("Experiment logged successfully!")
