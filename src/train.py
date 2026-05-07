import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Cảnh báo lệch lạc dữ liệu
    print("\n--- Data Distribution Analysis ---")
    dist = y_train.value_counts(normalize=True)
    for cls, ratio in dist.items():
        print(f"Class {cls}: {ratio:.2%}")
        if ratio < 0.10:
            print(f"WARNING: Class {cls} is underrepresented (<10%)!")

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen RandomForestClassifier
        # Goi y: su dung random_state=42 de dam bao tinh tai tao
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        # Log distribution to metrics for reference
        for cls, ratio in dist.items():
            mlflow.log_metric(f"ratio_class_{cls}", ratio)
        
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print("\n--- Model Performance ---")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 3: Báo cáo hiệu suất tự động (Precision/Recall per class)
        from sklearn.metrics import classification_report
        report_text = classification_report(y_eval, preds)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w") as f:
            f.write(report_text)
        print("\nClassification Report saved to outputs/report.txt")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc, 
                "f1_score": f1,
                "distribution": dist.to_dict()
            }, f)

        # TODO 9: Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len GCS o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
