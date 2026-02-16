import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import warnings

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
from uvicorn import run as app_run

from src.pipeline.train_pipeline import TrainPipeline

warnings.filterwarnings("ignore")

APP_HOST = "0.0.0.0"
APP_PORT = 8000

# ---------------------------------------------------------
# AUTO LOAD LATEST TRAINED MODEL
# ---------------------------------------------------------

def get_latest_model_path():
    artifact_path = os.path.join("src", "artifact")

    folders = [
        os.path.join(artifact_path, folder)
        for folder in os.listdir(artifact_path)
        if os.path.isdir(os.path.join(artifact_path, folder))
        and folder != "logs"
    ]

    latest_folder = sorted(folders)[-1]

    model_path = os.path.join(
        latest_folder,
        "model_trainer",
        "trained_model",
        "model.pkl"
    )

    return model_path


MODEL_PATH = get_latest_model_path()
model = joblib.load(MODEL_PATH)

print(f"✅ Loaded model from: {MODEL_PATH}")

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# FORM CLASS
# ---------------------------------------------------------

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request

    async def get_customer_data(self):
        form = await self.request.form()
        return [
            form.get('Age'),
            form.get('Education'),
            form.get('Marital_Status'),
            form.get('Parental_Status'),
            form.get('Children'),
            form.get('Income'),
            form.get('Total_Spending'),
            form.get('Days_as_Customer'),
            form.get('Recency'),
            form.get('Wines'),
            form.get('Fruits'),
            form.get('Meat'),
            form.get('Fish'),
            form.get('Sweets'),
            form.get('Gold'),
            form.get('Web'),
            form.get('Catalog'),
            form.get('Store'),
            form.get('Discount_Purchases'),
            form.get('Total_Promo'),
            form.get('NumWebVisitsMonth'),
        ]


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/train")
async def train_model():
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return Response("✅ Training completed successfully.")
    except Exception as e:
        return Response(f"❌ Training error: {e}")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "customer.html",
        {"request": request, "context": "Fill the form"}
    )


@app.post("/")
async def predict(request: Request):
    try:
        form = DataForm(request)
        input_data = await form.get_customer_data()

        input_df = pd.DataFrame(
            [input_data],
            columns=model.preprocessing_object.feature_names_in_
        )

        prediction = model.predict(input_df)

        return templates.TemplateResponse(
            "customer.html",
            {"request": request, "context": int(prediction[0])}
        )

    except Exception as e:
        return {"status": False, "error": str(e)}


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
