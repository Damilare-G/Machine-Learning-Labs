"""Export all MLflow runs of the lab06 experiment to results/mlflow_runs.csv (evidence for grading)."""
import mlflow
import trainer
from config_utils import load_config, tracking_uri


def main():
    cfg = load_config(trainer.LAB / "config.yaml")
    mlflow.set_tracking_uri(tracking_uri(cfg, trainer.LAB))
    df = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    keep = [c for c in df.columns if c in ("run_id", "status", "start_time", "tags.mlflow.runName", "tags.study")
            or c.startswith(("params.", "metrics."))]
    out = trainer.LAB / "results" / "mlflow_runs.csv"
    df[keep].sort_values("start_time").to_csv(out, index=False)
    print(f"{len(df)} runs exported to {out}")
    print("Browse them with:  mlflow ui --backend-store-uri", tracking_uri(cfg, trainer.LAB))


if __name__ == "__main__":
    main()