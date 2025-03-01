
![Starting_soon_Screen_13](https://github.com/user-attachments/assets/d648675e-4f85-4215-abe4-46311e54b09a)

# Hackathon Core API Documentation

This document provides API documentation for interacting with the Weather API.

**Base URL:** `https://core-api-xurt.onrender.com`

## Endpoints

### 1. Fetch and Store Weather Data for a Location (`/weather`)

*   **Endpoint:** `/weather`
*   **Method:** `POST`
*   **Description:** This endpoint fetches weather data from the Open-Meteo API for a given location and stores it in a Supabase database. It also optionally stores network performance metrics if provided.

#### Request Body (`application/json`)

```json
{
  "lat": 40.7128,        
  "lon": -74.0060,      
  "user_id": "optional-uuid-string", 
  "forecast_days": 7,   
  "timezone": "auto",   
  "effective_type": "4g",
  "downlink": 10.5,      
  "rtt": 50              
}
```
### 2. Retrieve Stored Weather Data for a Location (`/weather/{location_id}`)
- **Endpoint:** `/weather/{location_id}`
- **Method:** `GET`
- **Description:** Retrieves weather data stored in the database for a specific location ID.

#### Path Parameters
- `location_id` (string): Required. The unique identifier of the location for which to retrieve weather data. This`location_id` is returned in the response of the `/weather` endpoint.

### 3. Predict Internet Outage Probability (Rule-Based) (`/predict-outage-rule-based`)

- **Endpoint:** `/predict-outage-rule-based`
- **Method:** `POST`
- **Description:** Predicts the probability of an internet outage at a given location based on weather conditions and network performance metrics using a rule-based system.

#### Request Body (`application/json`)

```json
{
  "lat": 40.7128,       
  "lon": -74.0060,       
  "effective_type": "4g",
  "downlink": 5.2,      
  "rtt": 150             
}
```
