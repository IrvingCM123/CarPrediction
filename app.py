from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='Insurance Premium Prediction')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained regression model
model = load(pathlib.Path('model/train-regression-v1.joblib'))

# Create a lookup table for 'Vehicle_Age' and 'Vehicle_Damage' mappings
vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
vehicle_damage_mapping = {'Yes': 1, 'No': 0}

class InputData(BaseModel):
    age: int = 44
    driving_license: int = 1
    region_code: float = 28.0
    previously_insured: int = 0
    policy_sales_channel: float = 26.0
    vintage: int = 217

class OutputData(BaseModel):
    annual_premium: float

@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    
    # Prepare the input for prediction
    input_features = [
        data.age,
        data.driving_license,
        data.region_code,
        data.previously_insured,
        data.policy_sales_channel,
        data.vintage
    ]

    # Make the prediction using the model
    result = model.predict([input_features])

    return OutputData(annual_premium=result[0])
