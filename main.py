from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoConfig, AutoModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

config = AutoConfig.from_pretrained("GodfreyOwino/NPK_prediction_model2", trust_remote_code=True)
model = AutoModel.from_pretrained("GodfreyOwino/NPK_prediction_model2", config=config, trust_remote_code=True)

class PredictionInput(BaseModel):
    crop_name: list
    target_yield: list
    field_size: list
    ph: list
    organic_carbon: list
    nitrogen: list
    phosphorus: list
    potassium: list
    soil_moisture: list

@app.post("/predict")
async def predict(input_data: PredictionInput):
    prediction = model(input_data.dict())
    return {"prediction": prediction}

@app.get("/")
async def root():
    return {"message": "NPK Prediction API"}