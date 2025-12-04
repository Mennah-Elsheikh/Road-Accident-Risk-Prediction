"""
Test script for the Accident Risk Prediction API
This script tests the API endpoints to ensure they're working correctly.
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

# Test data
test_conditions = {
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

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_root_endpoint():
    """Test the root endpoint"""
    print_section("Testing Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_endpoint():
    """Test the health check endpoint"""
    print_section("Testing Health Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_endpoint():
    """Test the single prediction endpoint"""
    print_section("Testing Single Prediction Endpoint")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_conditions
        )
        print(f"Status Code: {response.status_code}")
        print(f"\nInput Data:")
        print(json.dumps(test_conditions, indent=2))
        print(f"\nPrediction Result:")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        if response.status_code == 200:
            print(f"\n✅ Prediction successful!")
            print(f"   Risk Score: {result['accident_risk']}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Confidence: {result['confidence']}")
            return True
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_predict_endpoint():
    """Test the batch prediction endpoint"""
    print_section("Testing Batch Prediction Endpoint")
    
    # Create multiple test conditions
    batch_conditions = [
        test_conditions,
        {
            **test_conditions,
            "road_type": "highway",
            "speed_limit": 70,
            "weather": "clear"
        },
        {
            **test_conditions,
            "road_type": "rural",
            "lighting": "night",
            "num_reported_accidents": 3
        }
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_conditions
        )
        print(f"Status Code: {response.status_code}")
        print(f"\nNumber of predictions requested: {len(batch_conditions)}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"Number of predictions received: {len(results)}")
            print(f"\nBatch Prediction Results:")
            for i, result in enumerate(results, 1):
                print(f"\n  Prediction {i}:")
                print(f"    Risk Score: {result['accident_risk']}")
                print(f"    Risk Level: {result['risk_level']}")
                print(f"    Confidence: {result['confidence']}")
            print(f"\n✅ Batch prediction successful!")
            return True
        else:
            print(f"❌ Batch prediction failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_various_scenarios():
    """Test various risk scenarios"""
    print_section("Testing Various Risk Scenarios")
    
    scenarios = [
        {
            "name": "Low Risk - Clear Day, Urban Road",
            "data": {
                **test_conditions,
                "weather": "clear",
                "lighting": "daylight",
                "road_type": "urban",
                "num_reported_accidents": 0
            }
        },
        {
            "name": "High Risk - Foggy Night, Curved Highway",
            "data": {
                **test_conditions,
                "weather": "foggy",
                "lighting": "night",
                "road_type": "highway",
                "curvature": 0.95,
                "speed_limit": 70,
                "num_reported_accidents": 5
            }
        },
        {
            "name": "Moderate Risk - Rainy Evening, Rural Road",
            "data": {
                **test_conditions,
                "weather": "rainy",
                "lighting": "dim",
                "road_type": "rural",
                "time_of_day": "evening",
                "num_reported_accidents": 2
            }
        }
    ]
    
    try:
        for scenario in scenarios:
            print(f"\n📊 Scenario: {scenario['name']}")
            response = requests.post(
                f"{BASE_URL}/predict",
                json=scenario['data']
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Risk Score: {result['accident_risk']:.4f}")
                print(f"   Risk Level: {result['risk_level']}")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🚗"*30)
    print("  ACCIDENT RISK PREDICTION API - TEST SUITE")
    print("🚗"*30)
    
    print(f"\n📡 Testing API at: {BASE_URL}")
    print("⚠️  Make sure the API server is running!")
    print("   Run: python api.py")
    
    # Run tests
    results = {
        "Root Endpoint": test_root_endpoint(),
        "Health Endpoint": test_health_endpoint(),
        "Single Prediction": test_predict_endpoint(),
        "Batch Prediction": test_batch_predict_endpoint(),
        "Various Scenarios": test_various_scenarios()
    }
    
    # Summary
    print_section("Test Summary")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"{'='*60}\n")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the API server.")

if __name__ == "__main__":
    main()
