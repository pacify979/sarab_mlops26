"""Request/response schemas for the inference API.

The feature set mirrors what src/data_prep.py produces: rolling telemetry
statistics, trailing error counts, and static machine metadata. Field names
match the trained model's expected columns exactly.
"""
from pydantic import BaseModel, Field


class TelemetryFeatures(BaseModel):
    """One feature vector for a single machine at a single point in time."""

    # --- Rolling sensor statistics (short 3h + long 24h windows) ---
    volt_mean_3h: float
    volt_std_3h: float
    volt_mean_24h: float
    volt_std_24h: float
    rotate_mean_3h: float
    rotate_std_3h: float
    rotate_mean_24h: float
    rotate_std_24h: float
    pressure_mean_3h: float
    pressure_std_3h: float
    pressure_mean_24h: float
    pressure_std_24h: float
    vibration_mean_3h: float
    vibration_std_3h: float
    vibration_mean_24h: float
    vibration_std_24h: float

    # --- Trailing 24h error counts ---
    error1_count_24h: float
    error2_count_24h: float
    error3_count_24h: float
    error4_count_24h: float
    error5_count_24h: float

    # --- Static machine metadata ---
    model: str = Field(..., examples=["model3"])
    age: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "volt_mean_3h": 170.2, "volt_std_3h": 12.1,
                "volt_mean_24h": 169.8, "volt_std_24h": 14.9,
                "rotate_mean_3h": 449.5, "rotate_std_3h": 40.2,
                "rotate_mean_24h": 447.1, "rotate_std_24h": 48.0,
                "pressure_mean_3h": 98.9, "pressure_std_3h": 8.4,
                "pressure_mean_24h": 100.1, "pressure_std_24h": 10.2,
                "vibration_mean_3h": 40.1, "vibration_std_3h": 4.2,
                "vibration_mean_24h": 40.4, "vibration_std_24h": 5.1,
                "error1_count_24h": 0, "error2_count_24h": 0,
                "error3_count_24h": 0, "error4_count_24h": 0,
                "error5_count_24h": 0,
                "model": "model3", "age": 18,
            }
        }
    }


class PredictionResponse(BaseModel):
    failure_probability: float = Field(..., description="P(component fails within 24h)")
    failure_predicted: bool = Field(..., description="probability >= decision threshold")
    threshold: float
