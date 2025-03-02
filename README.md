
![Starting_soon_Screen_13](https://github.com/user-attachments/assets/d648675e-4f85-4215-abe4-46311e54b09a)

# Hackathon Core API Documentation

This document provides API documentation for interacting with the Lonelist Team Core API.

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

### 4. Network Problem Diagnosis Chatbot (`/chatbot`)
- **Endpoint:** `/chatbot`
- **Method:** `POST`
- **Description:** This endpoint provides a chatbot interface powered by `DeepSeek R1` to assist with network problem diagnosis. It takes a user's text prompt and returns a chatbot response aimed at troubleshooting network issues.
#### Request Body (`application/json`)
```json
{
    "prompt": "We have a connection problem in our hospital, the network is very slow."
}
```
#### Response Body (`application/json`)
```json
{
    "response": "Thanks for reaching out! I understand you are experiencing network issues..."
}
```
### 5. Retrieve Coursework Document (`/documents/{document_id}`)

- **Endpoint:** `/documents/{document_id}`
- **Method:** `GET`
- **Description:** Retrieves a coursework document file from Supabase Storage based on its unique document ID. The document is returned as a file download.

#### Path Parameters

- `document_id` (UUID): **Required.** The unique identifier (UUID) of the coursework document to retrieve.

#### Response

- **Response Type:** File Download (e.g., application/pdf, application/vnd.openxmlformats-officedocument.wordprocessingml.document, text/plain, etc.)

- **Headers:**
- `Content-Type`:  Indicates the MIME type of the document (e.g., `application/pdf`).
- `Content-Disposition`:  Set to `attachment;filename="{document_filename}"` to suggest a filename for download.

- **Status Codes:**
- `200 OK`: Document retrieved successfully. The response body is the document file content.
- `404 Not Found`: Document with the given `document_id` was not found.
- `500 Internal Server Error`:  An error occurred during document retrieval (e.g., database error, storage error).

### 6. Retrieve Top 5 Prioritized Coursework Documents (`/documents/high-priority-docs`)

- **Endpoint:** `/documents/high-priority-docs`
- **Method:** `GET`
- **Description:** Retrieves the top 5 coursework documents from Supabase Storage, prioritized based on urgency (keyword "urgent" in filename), file size (smaller files prioritized), and upload date (older files prioritized).  Returns a list of documents with their metadata, priority level, and downloadable file content.

#### Query Parameters

- `downlink` (float, optional): Network downlink speed in Mbps. If provided, the response will include an estimated download time for each document based on this speed. If not provided, `estimated_download_time_seconds` will be `null` for each document.

#### Response Body (`application/json`)

```json
[
  {
    "document_id": "uuid-of-document-1",
    "filename": "urgent_assignment_older.pdf",
    "title": "Urgent Older Assignment",
    "uploaded_at": "2024-01-15T10:00:00Z",
    "posted_by": "uuid-of-user-1",
    "priority": "high",
    "content_type": "application/pdf",
    "content": "Base64 encoded file content..." 
  },
  {
    "document_id": "uuid-of-document-2",
    "filename": "quiz_smaller_size.docx",
    "title": "Smaller Size Quiz",
    "uploaded_at": "2024-02-20T14:30:00Z",
    "posted_by": "uuid-of-user-2",
    "priority": "medium",
    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "content": "Base64 encoded file content..."
  },
 { "... (up to 5 document entries) ..." },
]
```