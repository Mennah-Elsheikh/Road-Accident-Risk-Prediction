# 🚗 Accident Risk Prediction System

A comprehensive machine learning system for predicting accident risk based on road and environmental conditions. This project includes both a REST API and an interactive web interface.

## 📋 Features

### API (`api.py`)
- **FastAPI-based REST API** for accident risk predictions
- Single and batch prediction endpoints
- Comprehensive input validation
- Health check endpoint
- Interactive API documentation (Swagger UI)
- CORS support for cross-origin requests

### Streamlit App (`streamlit_app.py`)
- **Interactive web interface** for predictions
- Real-time risk visualization with gauge charts
- Risk categorization (Very Low, Low, Moderate, High, Very High)
- Safety recommendations based on conditions
- Input parameter summary
- Download prediction results as JSON

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd "d:/College Projects/AI2"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the API

Start the FastAPI server:
```bash
python api.py
```

Or using uvicorn directly:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Running the Streamlit App

Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The web interface will automatically open in your browser at http://localhost:8501

## 📡 API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint

### `POST /predict`
Predict accident risk for a single set of conditions

**Request Body:**
```json
{
  "road_type": "urban",
  "num_lanes": 2,
  "curvature": 0.06,
  "speed_limit": 35,
  "lighting": "daylight",
  "weather": "rainy",
  "road_signs_present": false,
  "public_road": true,
  "time_of_day": "afternoon",
  "holiday": false,
  "school_season": true,
  "num_reported_accidents": 1
}
```

**Response:**
```json
{
  "accident_risk": 0.1234,
  "risk_level": "Low",
  "confidence": "High"
}
```

### `POST /predict/batch`
Predict accident risk for multiple conditions at once

## 🔧 Input Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `road_type` | string | urban, rural, highway | Type of road |
| `num_lanes` | integer | 1-4 | Number of lanes |
| `curvature` | float | 0.0-1.0 | Road curvature (0=straight, 1=very curved) |
| `speed_limit` | integer | 25, 35, 45, 60, 70 | Speed limit in mph |
| `lighting` | string | daylight, dim, night | Lighting conditions |
| `weather` | string | clear, rainy, foggy | Weather conditions |
| `road_signs_present` | boolean | true/false | Presence of road signs |
| `public_road` | boolean | true/false | Whether it's a public road |
| `time_of_day` | string | morning, afternoon, evening | Time of day |
| `holiday` | boolean | true/false | Whether it's a holiday |
| `school_season` | boolean | true/false | Whether it's school season |
| `num_reported_accidents` | integer | 0+ | Previously reported accidents |

## 📊 Risk Levels

| Risk Score | Risk Level | Color | Description |
|------------|------------|-------|-------------|
| 0.0 - 0.2 | Very Low | Green | Very safe conditions |
| 0.2 - 0.35 | Low | Light Green | Generally safe conditions |
| 0.35 - 0.5 | Moderate | Yellow | Caution required |
| 0.5 - 0.7 | High | Orange | Hazardous conditions |
| 0.7 - 1.0 | Very High | Red | Very dangerous conditions |

## 🧪 Testing the API

### Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "road_type": "urban",
    "num_lanes": 2,
    "curvature": 0.06,
    "speed_limit": 35,
    "lighting": "daylight",
    "weather": "rainy",
    "road_signs_present": false,
    "public_road": true,
    "time_of_day": "afternoon",
    "holiday": false,
    "school_season": true,
    "num_reported_accidents": 1
  }'
```

### Using Python:
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "road_type": "urban",
    "num_lanes": 2,
    "curvature": 0.06,
    "speed_limit": 35,
    "lighting": "daylight",
    "weather": "rainy",
    "road_signs_present": False,
    "public_road": True,
    "time_of_day": "afternoon",
    "holiday": False,
    "school_season": True,
    "num_reported_accidents": 1
}

response = requests.post(url, json=data)
print(response.json())
```

## 📁 Project Structure

```
AI2/
├── api.py                  # FastAPI application
├── streamlit_app.py        # Streamlit web interface
├── EDA.ipynb              # Exploratory Data Analysis & Model Training
├── best_model.joblib      # Trained ML model
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/                 # Dataset directory
    ├── train.csv
    └── test.csv
```

## 🛠️ Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **Streamlit**: Interactive web application framework
- **Scikit-learn**: Machine learning library
- **XGBoost/LightGBM/CatBoost**: Gradient boosting frameworks
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Pydantic**: Data validation

## 📝 Model Information

The model is trained on road accident data with the following features:
- Road characteristics (type, lanes, curvature, speed limit)
- Environmental conditions (lighting, weather, time of day)
- Road features (signs, public/private)
- Contextual information (holiday, school season, previous accidents)

The model predicts accident risk as a continuous value between 0.0 (no risk) and 1.0 (maximum risk).

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

College Project - AI2

---

**Note**: Make sure the `best_model.joblib` file is present in the project directory before running the applications.
