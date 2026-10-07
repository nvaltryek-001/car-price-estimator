from pathlib import Path
from typing import Literal
import json
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "car_data.csv"
MODEL_PATH = ROOT / "ml" / "random_forest_model.joblib"
LINEAR_PATH = ROOT / "ml" / "linear_regression_model.joblib"
METRICS_PATH = ROOT / "ml" / "metrics.json"

app = FastAPI(title="Team 12 Used Car Price Estimator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

df = pd.read_csv(DATA_PATH)
rf_model = joblib.load(MODEL_PATH)
linear_model = joblib.load(LINEAR_PATH)
metrics = json.loads(METRICS_PATH.read_text())
CURRENT_YEAR = 2026

class PredictionRequest(BaseModel):
    year: int = Field(ge=1990, le=2030)
    present_price: float = Field(gt=0, le=200)
    kms_driven: int = Field(ge=0, le=1000000)
    fuel_type: Literal["Petrol", "Diesel", "CNG"]
    seller_type: Literal["Dealer", "Individual"]
    transmission: Literal["Manual", "Automatic"]
    owner: int = Field(ge=0, le=5)

@app.get("/api/health")
def health():
    return {"status":"ok", "dataset_rows": int(len(df)), "model":"Random Forest Regressor"}

@app.get("/api/model-info")
def model_info():
    return {"metrics": metrics, "target":"Selling_Price", "features":["Year","Age","Present_Price","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner"]}

@app.get("/api/eda")
def eda():
    clean=df.drop_duplicates().copy()
    clean['Age']=CURRENT_YEAR-clean['Year']
    fuel=(clean.groupby('Fuel_Type')['Selling_Price'].mean().round(2)).to_dict()
    transmission=(clean.groupby('Transmission')['Selling_Price'].mean().round(2)).to_dict()
    seller=(clean.groupby('Seller_Type')['Selling_Price'].mean().round(2)).to_dict()
    year=clean.groupby('Year').agg(avg_price=('Selling_Price','mean'),count=('Selling_Price','size')).reset_index()
    price_hist,bins=np.histogram(clean['Selling_Price'],bins=8)
    hist=[{"range":f"{bins[i]:.1f}-{bins[i+1]:.1f}","count":int(price_hist[i])} for i in range(len(price_hist))]
    corr=clean[['Year','Selling_Price','Present_Price','Kms_Driven','Owner','Age']].corr()['Selling_Price'].drop('Selling_Price').round(3).to_dict()
    return {"rows_raw":int(len(df)),"rows_clean":int(len(clean)),"duplicates_removed":int(len(df)-len(clean)),"fuel_avg":fuel,"transmission_avg":transmission,"seller_avg":seller,"year_trend":year.to_dict(orient='records'),"price_hist":hist,"correlation":corr}

@app.post("/api/predict")
def predict(req: PredictionRequest):
    age=max(0,CURRENT_YEAR-req.year)
    payload=pd.DataFrame([{
        'Year':req.year,'Age':age,'Present_Price':req.present_price,'Kms_Driven':req.kms_driven,
        'Fuel_Type':req.fuel_type,'Seller_Type':req.seller_type,'Transmission':req.transmission,'Owner':req.owner
    }])
    rf=float(rf_model.predict(payload)[0])
    lr=float(linear_model.predict(payload)[0])
    rf=max(0,rf); lr=max(0,lr)
    return {"random_forest_price_lakh":round(rf,2),"linear_regression_price_lakh":round(lr,2),"recommended_model":"Random Forest","currency":"INR","unit":"lakh"}
