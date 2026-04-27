from pydantic import BaseModel
from typing import Optional

class CropRecommendRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
