from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# Load the trained pipeline
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

# Define input schema
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Lead Scoring API is running"}

@app.post("/predict")
def predict(lead: Lead):
    # Convert input to dictionary for the pipeline
    X = [lead.dict()]
    prob = model.predict_proba(X)[0, 1]  # probability of conversion
    return {"conversion_probability": round(float(prob), 3)}
