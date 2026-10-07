# Team 12 — Used Car Price Estimator

**Members**
- DARSHAN H N — 1VA23CD025
- LIKHITH M — 1VA23CD044
- NIKESH — 1VA23CD058

## Topic
Used Car Price Estimation using Regression / Random Forest

## What is included
- React + Vite web application
- FastAPI machine-learning API
- Linear Regression baseline
- Random Forest Regression main model
- Data preprocessing: duplicate removal, missing-value handling through pipeline, categorical encoding, age feature
- EDA dashboard: fuel, transmission and year trends
- Model metrics: MAE, RMSE and R²
- Price prediction form
- Vercel monorepo configuration

## Dataset
`data/car_data.csv` follows the common CarDekho used-car schema: Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission and Owner. The included CSV is a deterministic educational dataset generated for this project so the ZIP is self-contained. **For the final college submission, replace it with the exact faculty-approved CarDekho CSV if your faculty has provided one.** The model will retrain only if you run the training script again.

## Local run
### Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

For local development, Vite should proxy `/api` to `http://localhost:8000` if you add a proxy in `vite.config.js`, or temporarily change the API URL. The deployed Vercel setup uses the `/api` rewrite.

## Deployment
This project includes `vercel.json` for a Vite frontend + FastAPI backend service layout. Vercel documents this monorepo pattern and FastAPI deployment.

1. Upload the ZIP to GitHub from your phone.
2. Import the GitHub repository into Vercel.
3. Deploy using the included `vercel.json`.
4. Test `/api/health` and the home page.

## Academic workflow
Dataset → preprocessing → EDA → train/test split → Linear Regression + Random Forest → evaluation → price prediction → web application.
