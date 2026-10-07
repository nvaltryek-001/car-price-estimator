from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"car_data.csv"
OUT=ROOT/"ml"
CURRENT_YEAR=2026
df=pd.read_csv(DATA).drop_duplicates().copy()
df["Age"]=CURRENT_YEAR-df["Year"]
X=df.drop(columns=["Selling_Price","Car_Name"])
y=df["Selling_Price"]
cat=["Fuel_Type","Seller_Type","Transmission"]
num=["Year","Age","Present_Price","Kms_Driven","Owner"]
pre=ColumnTransformer([
 ("num",SimpleImputer(strategy="median"),num),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
results={}
for name,estimator,file in [("Linear Regression",LinearRegression(),"linear_regression_model.joblib"),("Random Forest",RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1),"random_forest_model.joblib")]:
 pipe=Pipeline([('preprocessor',pre),('model',estimator)])
 pipe.fit(Xtr,ytr)
 pred=pipe.predict(Xte)
 results[name]={"mae":round(float(mean_absolute_error(yte,pred)),3),"rmse":round(float(mean_squared_error(yte,pred)**.5),3),"r2":round(float(r2_score(yte,pred)),4)}
 joblib.dump(pipe,OUT/file)
(OUT/"metrics.json").write_text(json.dumps(results,indent=2))
print(results)
