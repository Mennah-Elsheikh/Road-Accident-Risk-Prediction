# 🚀 Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Applications

### Option 1: Streamlit Web App (Recommended for beginners)

```bash
streamlit run streamlit_app.py
```

- Opens automatically in your browser at http://localhost:8501
- Interactive UI with sliders and dropdowns
- Visual risk gauge and recommendations
- No coding required!

### Option 2: FastAPI REST API

```bash
python api.py
```

- API available at http://localhost:8000
- Interactive docs at http://localhost:8000/docs
- Use for integration with other applications

## Testing the API

After starting the API server, run the test script in a new terminal:

```bash
python test_api.py
```

## Example API Usage

### Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"road_type\":\"urban\",\"num_lanes\":2,\"curvature\":0.06,\"speed_limit\":35,\"lighting\":\"daylight\",\"weather\":\"rainy\",\"road_signs_present\":false,\"public_road\":true,\"time_of_day\":\"afternoon\",\"holiday\":false,\"school_season\":true,\"num_reported_accidents\":1}"
```

### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
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
)

print(response.json())
```

## Troubleshooting

### Model not found error:
- Make sure `best_model.joblib` exists in the project directory
- Check that the file path is correct

### Port already in use:
- **Streamlit**: Use `streamlit run streamlit_app.py --server.port 8502`
- **API**: Change port in `api.py` or use `uvicorn api:app --port 8001`

### Import errors:
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.8 or higher

## Next Steps

1. ✅ Start with the Streamlit app to understand the model
2. ✅ Explore the API documentation at http://localhost:8000/docs
3. ✅ Run the test script to see various scenarios
4. ✅ Integrate the API into your own applications

## Files Overview

- `streamlit_app.py` - Interactive web interface
- `api.py` - REST API server
- `test_api.py` - API testing script
- `requirements.txt` - Python dependencies
- `best_model.joblib` - Trained ML model
- `EDA.ipynb` - Model training notebook

---

**Need help?** Check the full README.md for detailed documentation.
