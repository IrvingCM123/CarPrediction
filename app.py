from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Importa CORSMiddleware


app = FastAPI(title = 'Heart Disease Prediction')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load(pathlib.Path('model/true_car_listings-v1.joblib'))

class InputData(BaseModel):
    Year: int = 2014
    Mileage: int = 35725
    City: str = 'El Paso'
    State: str = 'TX'
    Vin: str = '19VDE2E53EE000083'
    Make: str = 'Acura'
    Model: str = 'ILX6-Speed'


class OutputData(BaseModel):
    score:float = 8995
 
@app.post('/score', response_model = OutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict_proba(model_input)[:,-1]

    return {'score':result}


