from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class EcowittPayload(BaseModel):
    """Raw Ecowitt form fields after they have been lifted into a dict by api.py."""

    PASSKEY:       Optional[str]   = None
    dateutc:       Optional[str]   = None
    tempf:         Optional[float] = None
    humidity:      Optional[float] = None
    baromrelin:    Optional[float] = None
    baromabsin:    Optional[float] = None
    winddir:       Optional[float] = None
    windspeedmph:  Optional[float] = None
    windgustmph:   Optional[float] = None
    rainratein:    Optional[float] = None
    hourlyrainin:  Optional[float] = None
    dailyrainin:   Optional[float] = None
    solarradiation: Optional[float] = None
    uv:            Optional[float] = None


# ---------------------------------------------------------------------------
# Internal / storage
# ---------------------------------------------------------------------------

class ObservationCreate(BaseModel):
    """Metric-converted observation ready to be written to the database."""

    passkey:          Optional[str]   = None
    dateutc:          Optional[str]   = None
    temp_c:           Optional[float] = Field(None, description="Air temperature (°C)")
    humidity:         Optional[float] = Field(None, ge=0, le=100, description="Relative humidity (%)")
    pressure_rel_hpa: Optional[float] = Field(None, description="Relative pressure (hPa)")
    pressure_abs_hpa: Optional[float] = Field(None, description="Absolute pressure (hPa)")
    wind_speed_ms:    Optional[float] = Field(None, ge=0, description="Wind speed (m/s)")
    wind_dir_deg:     Optional[float] = Field(None, ge=0, le=360, description="Wind direction (°)")
    wind_gust_ms:     Optional[float] = Field(None, ge=0, description="Wind gust (m/s)")
    rain_rate_mm_hr:  Optional[float] = Field(None, ge=0, description="Rain rate (mm/hr)")
    rain_hourly_mm:   Optional[float] = Field(None, ge=0, description="Hourly rain accumulation (mm)")
    rain_daily_mm:    Optional[float] = Field(None, ge=0, description="Daily rain accumulation (mm)")
    solar_radiation:  Optional[float] = Field(None, ge=0, description="Solar radiation (W/m²)")
    uv_index:         Optional[float] = Field(None, ge=0, description="UV index")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class IngestionResponse(BaseModel):
    status: str  # "ok" | "skipped"


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class StationResponse(BaseModel):
    station_id:  str
    name:        Optional[str]  = None
    lat:         Optional[float] = None
    lon:         Optional[float] = None
    elevation_m: Optional[float] = None
    timezone:    Optional[str]  = None
    created_at:  Optional[datetime] = None

    model_config = {"from_attributes": True}


class LeadErrorResponse(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None
    bias: Optional[float] = None
    n: int = 0


class ForecastBiasResponse(BaseModel):
    temp_bias: Optional[float] = None
    precip_bias: Optional[float] = None
    wind_speed_bias: Optional[float] = None
    pressure_bias: Optional[float] = None
    n: int = 0


class BaselineResponse(BaseModel):
    station_id: str
    days: int
    baseline_errors: dict[int, LeadErrorResponse]
    forecast_bias: ForecastBiasResponse


class ObservationResponse(BaseModel):
    time:          datetime
    station_id:    str
    temp_c:        Optional[float] = None
    humidity_pct:  Optional[float] = None
    pressure_hpa:  Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_dir_deg:  Optional[float] = None
    wind_gust_ms:  Optional[float] = None
    rain_mm_1h:          Optional[float] = None
    rain_mm_daily_total: Optional[float] = None
    rain_rate_mm_hr:     Optional[float] = None
    solar_wm2:           Optional[float] = None
    uv_index:            Optional[float] = None

    model_config = {"from_attributes": True}
