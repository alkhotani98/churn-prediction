import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("models/xgb_model.pkl")

app = FastAPI()


class Features(BaseModel):
    songs_played: int
    thumbs_up: int
    thumbs_down: int
    add_playlist: int
    days_active: int
    recency_days: int


@app.post("/predict")
def predict(f: Features):
    X = np.array(
        [
            [
                f.songs_played,
                f.thumbs_up,
                f.thumbs_down,
                f.add_playlist,
                f.days_active,
                f.recency_days,
            ]
        ]
    )
    probability = float(model.predict_proba(X)[0, 1])
    return {"churn_probability": probability}
