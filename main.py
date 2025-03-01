from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import httpx
import os
from datetime import datetime
from supabase import create_client, Client
import logging
from typing import Optional, Dict, Any
from uuid import UUID
from huggingface_hub import InferenceClient  # Import InferenceClient

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Add Hugging Face API Key

# Check for required environment variables
if not SUPABASE_URL:
    raise Exception("SUPABASE_URL environment variable is not set.")
if not SUPABASE_KEY:
    raise Exception("SUPABASE_KEY environment variable is not set.")
if not HUGGINGFACE_API_KEY:
    raise Exception("HUGGINGFACE_API_KEY environment variable is not set.") # Check for HF API Key


# Initialize FastAPI app
app = FastAPI(title="Weather API Backend")

# Initialize Hugging Face Inference Client
try:
    hf_client = InferenceClient(
        provider="together",
        api_key=HUGGINGFACE_API_KEY
    )
    logger.info("Hugging Face Inference Client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Hugging Face Inference Client: {e}")
    # It's crucial to handle initialization errors gracefully.
    # Consider raising an exception or disabling chatbot functionality if initialization fails.
    # For now, we'll log the error, but the app will still try to start (chatbot endpoint might not work).


# Simplified model
class LocationCoordinates(BaseModel):
    lat: float = Field(..., description="Latitude of the location")
    lon: float = Field(..., description="Longitude of the location")
    user_id: Optional[UUID] = Field(None, description="User identifier for tracking")
    forecast_days: Optional[int] = Field(7, description="Number of forecast days (1-16)")
    timezone: Optional[str] = Field("auto", description="Timezone for weather data")
    effective_type: Optional[str] = Field(None, description="Effective connection type (e.g., '4g', '3g')")
    downlink: Optional[float] = Field(None, description="Estimated bandwidth in Mbps")
    rtt: Optional[int] = Field(None, description="Round-trip time in milliseconds")

# Chatbot Request Model
class ChatbotRequest(BaseModel):
    prompt: str = Field(..., description="User's message prompt for the chatbot")


# Dependency for Supabase client
def get_supabase() -> Client:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

# Helper function to convert values for database compatibility (rest of your existing functions remain the same)
def convert_value_for_db(key: str, value: Any) -> Any:
    """Convert values to appropriate types for database storage"""
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    integer_fields = [
        "pressure", "humidity", "visibility", "wind_direction",
        "weather_code", "is_day", "cloud_cover", "cloud_cover_low",
        "cloud_cover_mid", "cloud_cover_high", "precipitation_hours",
        "precipitation_probability_max"
    ]
    if key in integer_fields and isinstance(value, (float, str)):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {key}={value} to integer, using original value")
            return value
    return value

def sanitize_data_for_db(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize all values in a dictionary for database compatibility"""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = sanitize_data_for_db(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_data_for_db(item) if isinstance(item, dict)
                else convert_value_for_db(key, item)
                for item in value
            ]
        else:
            result[key] = convert_value_for_db(key, value)
    return result

def generate_location_id(lat: float, lon: float, user_id: Optional[UUID] = None) -> str:
    """Generate a unique identifier for a location"""
    if user_id:
        return f"{str(user_id)}_{lat:.4f}_{lon:.4f}"
    return f"{lat:.4f}_{lon:.4f}"

async def fetch_weather_data(lat: float, lon: float, forecast_days: int = 7, timezone: str = "auto") -> Dict[str, Any]:
    """Fetch weather data from Open-Meteo API"""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "forecast_days": forecast_days,
        "current": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "is_day", "weather_code", "surface_pressure", "wind_speed_10m",
            "wind_direction_10m", "visibility", "uv_index"
        ],
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
            "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "precipitation", "snowfall", "snow_depth", "weather_code", "visibility", "is_day"
        ],
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
            "apparent_temperature_min", "precipitation_sum", "rain_sum", "showers_sum",
            "snowfall_sum", "precipitation_hours", "precipitation_probability_max",
            "weather_code", "sunrise", "sunset", "wind_speed_10m_max",
            "wind_gusts_10m_max", "wind_direction_10m_dominant", "uv_index_max"
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(base_url, params=params)
        if response.status_code != 200:
            logger.error(f"Open-Meteo API error: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Error fetching weather data")
        return response.json()

async def fetch_weather_data_for_rules(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch only relevant weather data for rule-based prediction from Open-Meteo API"""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": 1, # Fetch only for today
        "current": [
            "weather_code", "wind_speed_10m"
        ],
        "daily": [
            "weather_code", "wind_speed_10m_max", "wind_gusts_10m_max", "precipitation_sum", "precipitation_probability_max"
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(base_url, params=params)
        if response.status_code != 200:
            logger.error(f"Open-Meteo API error: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Error fetching weather data for rules")
        return response.json()

def format_weather_data(raw_data: Dict[str, Any], location_id: str) -> Dict[str, Any]:
    """Format and categorize weather data for database storage"""
    current = None
    if "current" in raw_data:
        current = {
            "dt": int(datetime.fromisoformat(raw_data["current"]["time"]).timestamp()),
            "temp": raw_data["current"]["temperature_2m"],
            "feels_like": raw_data["current"]["apparent_temperature"],
            "humidity": raw_data["current"]["relative_humidity_2m"],
            "pressure": raw_data["current"]["surface_pressure"],
            "wind_speed": raw_data["current"]["wind_speed_10m"],
            "wind_direction": raw_data["current"]["wind_direction_10m"],
            "weather_code": raw_data["current"]["weather_code"],
            "is_day": raw_data["current"]["is_day"],
            "visibility": raw_data["current"].get("visibility"),
            "uv_index": raw_data["current"].get("uv_index"),
            "location_id": location_id
        }
    hourly = []
    for i, time_str in enumerate(raw_data["hourly"]["time"]):
        dt = int(datetime.fromisoformat(time_str).timestamp())
        hourly_data = {
            "dt": dt,
            "temp": raw_data["hourly"]["temperature_2m"][i],
            "feels_like": raw_data["hourly"]["apparent_temperature"][i],
            "humidity": raw_data["hourly"]["relative_humidity_2m"][i],
            "pressure": raw_data["hourly"]["surface_pressure"][i],
            "wind_speed": raw_data["hourly"]["wind_speed_10m"][i],
            "wind_direction": raw_data["hourly"]["wind_direction_10m"][i],
            "wind_gusts": raw_data["hourly"].get("wind_gusts_10m", [None])[i],
            "precipitation": raw_data["hourly"]["precipitation"][i],
            "snowfall": raw_data["hourly"]["snowfall"][i],
            "weather_code": raw_data["hourly"]["weather_code"][i],
            "cloud_cover": raw_data["hourly"]["cloud_cover"][i],
            "cloud_cover_low": raw_data["hourly"].get("cloud_cover_low", [None])[i],
            "cloud_cover_mid": raw_data["hourly"].get("cloud_cover_mid", [None])[i],
            "cloud_cover_high": raw_data["hourly"].get("cloud_cover_high", [None])[i],
            "visibility": raw_data["hourly"].get("visibility", [None])[i],
            "is_day": raw_data["hourly"]["is_day"][i],
            "snow_depth": raw_data["hourly"].get("snow_depth", [None])[i],
            "location_id": location_id
        }
        hourly.append(hourly_data)
    daily = []
    if "daily" in raw_data:
        for i, time_str in enumerate(raw_data["daily"]["time"]):
            dt = int(datetime.fromisoformat(time_str).timestamp())
            daily_data = {
                "dt": dt,
                "temperature_max": raw_data["daily"]["temperature_2m_max"][i],
                "temperature_min": raw_data["daily"]["temperature_2m_min"][i],
                "apparent_temperature_max": raw_data["daily"]["apparent_temperature_max"][i],
                "apparent_temperature_min": raw_data["daily"]["apparent_temperature_min"][i],
                "precipitation_sum": raw_data["daily"]["precipitation_sum"][i],
                "rain_sum": raw_data["daily"]["rain_sum"][i],
                "showers_sum": raw_data["daily"]["showers_sum"][i],
                "snowfall_sum": raw_data["daily"]["snowfall_sum"][i],
                "precipitation_hours": raw_data["daily"]["precipitation_hours"][i],
                "precipitation_probability_max": raw_data["daily"].get("precipitation_probability_max", [None])[i],
                "weather_code": raw_data["daily"]["weather_code"][i],
                "sunrise": int(datetime.fromisoformat(raw_data["daily"]["sunrise"][i]).timestamp()),
                "sunset": int(datetime.fromisoformat(raw_data["daily"]["sunset"][i]).timestamp()),
                "wind_speed_max": raw_data["daily"]["wind_speed_10m_max"][i],
                "wind_gusts_max": raw_data["daily"]["wind_gusts_10m_max"][i],
                "wind_direction_dominant": raw_data["daily"]["wind_direction_10m_dominant"][i],
                "uv_index_max": raw_data["daily"].get("uv_index_max", [None])[i],
                "location_id": location_id
            }
            daily.append(daily_data)
    return {
        "current": current,
        "hourly": hourly,
        "daily": daily
    }

async def store_weather_data(data: Dict[str, Any], supabase: Client) -> Dict[str, Any]:
    """Store formatted weather data in Supabase with type conversion"""
    results = {}
    try:
        sanitized_data = sanitize_data_for_db(data)
        location_result = supabase.table("weather_locations").upsert(sanitized_data["location"]).execute()
        results["location"] = location_result.data
        if sanitized_data["current"]:
            try:
                current_result = supabase.table("current_weather").upsert(sanitized_data["current"]).execute()
                results["current"] = current_result.data
            except Exception as e:
                logger.error(f"Error storing current weather: {str(e)}")
        batch_size = 50
        hourly_results = []
        for i in range(0, len(sanitized_data["hourly"]), batch_size):
            batch = sanitized_data["hourly"][i:i+batch_size]
            try:
                hourly_batch_result = supabase.table("hourly_weather").upsert(batch).execute()
                hourly_results.extend(hourly_batch_result.data)
            except Exception as e:
                logger.error(f"Error storing hourly batch {i//batch_size}: {str(e)}")
        results["hourly"] = hourly_results
        if sanitized_data["daily"]:
            try:
                daily_result = supabase.table("daily_weather").upsert(sanitized_data["daily"]).execute()
                results["daily"] = daily_result.data
            except Exception as e:
                logger.error(f"Error storing daily weather: {str(e)}")
        return results
    except Exception as e:
        logger.error(f"Error in store_weather_data: {str(e)}")
        raise e

async def predict_outage_probability_rule_based(coords: LocationCoordinates):
    """Predict outage probability based on weather and network data using rule-based logic."""
    weather_data = await fetch_weather_data_for_rules(coords.lat, coords.lon)
    network_metrics = {
        "downlink": coords.downlink,
        "rtt": coords.rtt
    }
    outage_probability = "Low" # Default
    daily_weather = weather_data.get("daily", {})
    current_weather = weather_data.get("current", {})
    wind_speed_max_list = daily_weather.get("wind_speed_10m_max", []) # Get list, default to empty list
    wind_gusts_max_list = daily_weather.get("wind_gusts_10m_max", []) # Get list, default to empty list
    wind_speed_max = wind_speed_max_list[0] if isinstance(wind_speed_max_list, list) and wind_speed_max_list else 0 # Safe access, default 0
    wind_gusts_max = wind_gusts_max_list[0] if isinstance(wind_gusts_max_list, list) and wind_gusts_max_list else 0 # Safe access, default 0
    if wind_speed_max >= 70 or wind_gusts_max >= 90:
        outage_probability = "High"
        return {"outage_probability": outage_probability, "reasoning": "Rule 1 Triggered: High Wind/Gusts"}
    heavy_precipitation_codes = [61, 63, 65, 66, 67, 71, 73, 75, 77, 82, 85, 86, 95, 96, 99] # Example codes for heavy rain/snow/storms - CHECK OPEN-METEO DOCS
    daily_weather_code_list = daily_weather.get("weather_code", [])
    current_weather_code_list_raw = current_weather.get("weather_code", []) # Get raw value, might be list or single value
    daily_weather_code = daily_weather_code_list[0] if isinstance(daily_weather_code_list, list) and daily_weather_code_list else 0
    current_weather_code = current_weather_code_list_raw[0] if isinstance(current_weather_code_list_raw, list) and current_weather_code_list_raw else current_weather_code_list_raw if isinstance(current_weather_code_list_raw, int) else 0 # Handle both list and int cases
    precipitation_sum_list = daily_weather.get("precipitation_sum", [])
    precipitation_sum = precipitation_sum_list[0] if isinstance(precipitation_sum_list, list) and precipitation_sum_list else 0 # Safe access, default 0
    if precipitation_sum >= 30 or daily_weather_code in heavy_precipitation_codes or current_weather_code in heavy_precipitation_codes:
        outage_probability = "Medium to High"
        if outage_probability != "High": # Don't override "High" from Rule 1
            return {"outage_probability": outage_probability, "reasoning": "Rule 2 Triggered: Heavy Precipitation"}
    if (wind_speed_max >= 50 or wind_gusts_max >= 60) and precipitation_sum >= 10:
        outage_probability = "Medium"
        if outage_probability != "High" and outage_probability != "Medium to High": # Don't override higher risks
            return {"outage_probability": outage_probability, "reasoning": "Rule 3 Triggered: Moderate Wind + Precipitation"}
    if network_metrics["rtt"] is not None and network_metrics["downlink"] is not None: # Check for None values
        if network_metrics["rtt"] >= 200 and network_metrics["downlink"] <= 1:
            outage_probability = "Medium"
            if outage_probability != "High" and outage_probability != "Medium to High": # Don't override higher risks
                return {"outage_probability": outage_probability, "reasoning": "Rule 4 Triggered: Network Degradation"}
    if outage_probability != "Low": # If any weather risk is already Medium or High
        if network_metrics["rtt"] is not None and network_metrics["downlink"] is not None: # Check for None values
            if network_metrics["rtt"] >= 100 and network_metrics["downlink"] <= 5:
                outage_probability = "Medium to High" #Escalate to Medium to High if network is also degraded
                return {"outage_probability": outage_probability, "reasoning": "Rule 5 Triggered: Network Degraded + Weather Risk"}
    return {"outage_probability": outage_probability, "reasoning": "No specific rule triggered - Low Probability (Default)"}

@app.post("/weather", description="Fetch and store weather data for a location")
async def get_weather(
    coords: LocationCoordinates,
    supabase: Client = Depends(get_supabase)
):
    try:
        logger.info(f"Processing weather request for coordinates: {coords.lat}, {coords.lon}")
        location_id = generate_location_id(coords.lat, coords.lon, coords.user_id)
        logger.info(f"Generated location_id: {location_id}")
        logger.info("Fetching data from Open-Meteo...")
        weather_data = await fetch_weather_data(
            coords.lat,
            coords.lon,
            forecast_days=coords.forecast_days,
            timezone=coords.timezone
        )
        logger.info("Successfully fetched data from Open-Meteo")
        logger.info("Formatting data for storage...")
        formatted_data = format_weather_data(weather_data, location_id)
        logger.info("Data formatted successfully")
        formatted_data["location"] = {
            "id": location_id,
            "lat": coords.lat,
            "lon": coords.lon,
            "timezone": weather_data["timezone"],
            "timezone_abbreviation": weather_data["timezone_abbreviation"],
            "elevation": weather_data["elevation"],
            "user_id": coords.user_id,
            "created_at": datetime.now().isoformat()
        }
        logger.info("Storing data in Supabase...")
        storage_result = await store_weather_data(formatted_data, supabase)
        logger.info("Data stored successfully")
        if any([coords.effective_type, coords.downlink, coords.rtt]):
            network_data = {
                "user_id": str(coords.user_id) if coords.user_id else None,
                "location_id": location_id,
                "effective_type": coords.effective_type,
                "downlink": coords.downlink,
                "rtt": coords.rtt,
                "created_at": datetime.now().isoformat()
            }
            try:
                supabase.table("network_logs").insert(network_data).execute()
                logger.info("Network data stored successfully")
            except Exception as e:
                logger.error(f"Error storing network data: {str(e)}")
        success_data = {
            "message": "Weather data processing complete",
            "location_id": location_id,
            "data": {
                "current": "current" in storage_result,
                "hourly_count": len(storage_result.get("hourly", [])),
                "daily_count": len(storage_result.get("daily", []))
            }
        }
        return success_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing weather data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing weather data: {str(e)}")

@app.get("/weather/{location_id}", description="Get stored weather data for a location")
async def retrieve_weather(
    location_id: str,
    supabase: Client = Depends(get_supabase)
):
    try:
        location = supabase.table("weather_locations").select("*").eq("id", location_id).execute()
        if not location.data:
            raise HTTPException(status_code=404, detail="Location not found")
        current = supabase.table("current_weather").select("*").eq("location_id", location_id).order("dt.desc").limit(1).execute()
        hourly = supabase.table("hourly_weather").select("*").eq("location_id", location_id).order("dt.asc").execute()
        daily = supabase.table("daily_weather").select("*").eq("location_id", location_id).order("dt.asc").execute()
        return {
            "location": location.data[0],
            "current": current.data[0] if current.data else None,
            "hourly": hourly.data,
            "daily": daily.data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving weather data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving weather data: {str(e)}")

@app.post("/predict-outage-rule-based", description="Predict internet outage probability using rule-based logic")
async def get_outage_prediction_rule_based(coords: LocationCoordinates):
    """Endpoint to predict outage probability based on rule-based system."""
    try:
        prediction_result = await predict_outage_probability_rule_based(coords)
        return prediction_result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error predicting outage probability (rule-based): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error predicting outage probability: {str(e)}")

# Chatbot Endpoint
@app.post("/chatbot", description="Chatbot endpoint for network problem diagnosis")
async def chatbot_endpoint(request: ChatbotRequest):
    """Endpoint to interact with the DeepSeek R1 chatbot for network problem diagnosis."""
    try:
        if not hf_client: # Check if client initialized properly
            raise HTTPException(status_code=500, detail="Chatbot service unavailable. Inference Client initialization failed.")

        user_prompt = request.prompt

        # System prompt to guide DeepSeek R1 for network diagnosis - You can refine this further!
        system_prompt = """You are a highly intelligent AI assistant specializing in diagnosing and solving network problems in a hospital environment.
        Your goal is to help users troubleshoot network issues.
        Ask clarifying questions to understand the problem, consider any information provided, and suggest logical, step-by-step troubleshooting actions.
        Focus on accuracy and providing helpful, practical advice related to network connectivity, router issues, and common hospital network scenarios.
        When a user provides an image (or says they have), acknowledge it and ask them to describe visual details relevant to the network problem."""


        messages = [
            {"role": "system", "content": system_prompt}, # System prompt for context
            {"role": "user", "content": user_prompt}      # User's prompt
        ]

        completion = hf_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages,
            max_tokens=5000, # Increase max tokens for longer responses
            temperature=0.1, # Lower temperature for more deterministic responses
        )

        response_text = completion.choices[0].message.content
        return {"response": response_text}

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e: # Catch other exceptions and return as HTTP 500
        logger.error(f"Chatbot error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chatbot service error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)