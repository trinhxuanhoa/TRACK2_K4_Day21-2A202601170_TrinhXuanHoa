# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
REFERENCE_POS_RATIO = 0.248
DRIFT_TOLERANCE = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # Bonus 1: Ho tro Remote Tracking Server (DagsHub / Tu xa) neu co bien moi truong
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiem tra phan phoi du lieu (Data Drift Check)
    pos_ratio = float((y_train == 1).mean())
    print(f"[Bonus 5] Ty le lop duong (>50K) tap train: {pos_ratio*100:.2f}% (Tham chieu: {REFERENCE_POS_RATIO*100:.1f}%)")
    if abs(pos_ratio - REFERENCE_POS_RATIO) > DRIFT_TOLERANCE:
        print(f"CANH BAO (Data Drift): Ty le lop duong lech qua 5 diem phan tram so voi tham chieu {REFERENCE_POS_RATIO*100:.1f}%!")

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Du doan tren tap holdout tai nguong mac dinh 0.5
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds, pos_label=1, zero_division=0))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: Dieu chinh nguong quyet dinh (Decision Threshold Tuning)
        probs = model.predict_proba(X_eval)[:, 1]
        best_thresh = 0.5
        best_f1 = f1
        for thresh in np.arange(0.10, 0.95, 0.05):
            thresh = round(float(thresh), 2)
            t_preds = (probs >= thresh).astype(int)
            t_f1 = float(f1_score(y_eval, t_preds, pos_label=1, zero_division=0))
            if t_f1 > best_f1:
                best_f1 = t_f1
                best_thresh = thresh

        print(f"[Bonus 2] Nguong mac dinh (0.50): F1 = {f1:.4f} | Nguong toi uu ({best_thresh:.2f}): F1 = {best_f1:.4f}")

        # Bonus 3: Tinh toan Confusion Matrix va Precision / Recall chi tiet
        prec_0 = float(precision_score(y_eval, preds, pos_label=0, zero_division=0))
        rec_0 = float(recall_score(y_eval, preds, pos_label=0, zero_division=0))
        prec_1 = float(precision_score(y_eval, preds, pos_label=1, zero_division=0))
        rec_1 = float(recall_score(y_eval, preds, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_eval, preds)
        report_str = classification_report(y_eval, preds, target_names=["<=50K (0)", ">50K (1)"], digits=4)

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write("=== BAO CAO CHI TIET CLASSIFICATION REPORT (BONUS 3) ===\n\n")
            f.write("1. Confusion Matrix (Ma tran nham lan):\n")
            f.write(f"   TN (<=50K dung): {cm[0, 0]:<5} | FP (<=50K doan thanh >50K): {cm[0, 1]}\n")
            f.write(f"   FN (>50K bo sot): {cm[1, 0]:<5} | TP (>50K dung): {cm[1, 1]}\n\n")
            f.write("2. Precision va Recall tung lop:\n")
            f.write(f"   - Lop <=50K (0): Precision = {prec_0:.4f}, Recall = {rec_0:.4f}\n")
            f.write(f"   - Lop >50K  (1): Precision = {prec_1:.4f}, Recall = {rec_1:.4f}\n\n")
            f.write(f"3. Bang Classification Report tong hop:\n{report_str}\n\n")
            f.write("4. Ket qua quet nguong quyet dinh (Bonus 2):\n")
            f.write(f"   - Nguong mac dinh (0.50): F1 = {f1:.4f}\n")
            f.write(f"   - Nguong toi uu ({best_thresh:.2f}): F1 = {best_f1:.4f}\n\n")
            f.write("5. Kiem tra phan phoi du lieu (Bonus 5):\n")
            f.write(f"   - Ty le lop duong trong tap train: {pos_ratio*100:.2f}%\n")
            f.write(f"   - Ty le tham chieu: {REFERENCE_POS_RATIO*100:.1f}%\n")

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_class_ratio", pos_ratio)
        mlflow.log_metric("best_threshold", best_thresh)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.log_metric("precision_class_1", prec_1)
        mlflow.log_metric("recall_class_1", rec_1)
        mlflow.sklearn.log_model(model, "model")

        # In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # Luu metrics day du ra outputs/report.json
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "positive_class_ratio": round(pos_ratio, 4),
            "default_threshold": 0.5,
            "best_threshold": best_thresh,
            "best_f1_score": round(best_f1, 4),
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        # Luu mo hinh ra file models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)

