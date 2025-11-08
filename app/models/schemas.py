from pydantic import BaseModel
from typing import Optional, List

class AnalysisRequest(BaseModel):
    plant_type: Optional[str] = "general"

class DiseaseResponse(BaseModel):
    name: str
    confidence: float
    treatment: str
    prevention: str
    urgency: str  # low, medium, high

class AnalysisResponse(BaseModel):
    disease: DiseaseResponse
    weather_impact: str
    recommendation: str
    timestamp: str

class WeatherResponse(BaseModel):
    temperature: float
    humidity: int
    conditions: str
    recommendation: str
    location: str