from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "income-lab-bucket-trinhxuanhoa-2026")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tai file model.joblib tu cloud storage (AWS S3) ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    AWS credentials / IAM role de xac thuc (duoc dat trong systemd service).
    """
    # TODO 1-4: Tao boto3 S3 client va tai file model xuong may
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(ARTIFACT_BUCKET, MODEL_KEY, MODEL_PATH)
    print("Model da duoc tai xuong tu cloud storage (AWS S3).")


# download_model() duoc goi khi server khoi dong tren VM de nap model moi nhat tu S3
if os.environ.get("AUTO_DOWNLOAD_MODEL", "1") == "1":
    try:
        download_model()
    except Exception as e:
        print(f"Warning: Could not download model during init: {e}")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    pred = model.predict([req.features])[0]
    pred_int = int(pred)
    label = "thu_nhap_cao" if pred_int == 1 else "thu_nhap_thap"

    return {"prediction": pred_int, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

