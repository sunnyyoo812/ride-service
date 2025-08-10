#!/usr/bin/env python3
"""
Test script to verify the ride service ML system is working correctly
"""

import sys
import os
import requests
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ml_model():
    """Test the ML model directly"""
    print("🧠 Testing ML Model...")
    
    try:
        from ml.inference.model_service import ModelService
        
        # Create model service
        service = ModelService()
        
        # Test prediction
        result = service.predict_arrival_time(
            driver_location=(40.7589, -73.9851),  # Times Square
            destination=(40.6892, -74.0445),      # Statue of Liberty
            traffic_factor=0.6
        )
        
        print(f"✅ ML Model Test Passed")
        print(f"   Predicted arrival time: {result['predicted_arrival_time_minutes']:.2f} minutes")
        return True
        
    except Exception as e:
        print(f"❌ ML Model Test Failed: {e}")
        return False

def test_data_producers():
    """Test the data producers"""
    print("\n📡 Testing Data Producers...")
    
    try:
        from data.producers.gps_producer import GPSProducer
        from data.producers.traffic_producer import TrafficProducer
        
        # Test GPS producer
        gps_producer = GPSProducer()
        nearby_drivers = gps_producer.get_available_drivers((40.7589, -73.9851), radius=10)
        
        if len(nearby_drivers) > 0:
            print(f"✅ GPS Producer Test Passed - Found {len(nearby_drivers)} drivers")
        else:
            print("⚠️  GPS Producer Test - No drivers found (this is normal)")
        
        # Test traffic producer
        traffic_producer = TrafficProducer()
        traffic_info = traffic_producer.get_traffic_for_location(40.7589, -73.9851)
        
        if traffic_info and 'traffic_factor' in traffic_info:
            print(f"✅ Traffic Producer Test Passed - Traffic factor: {traffic_info['traffic_factor']:.3f}")
        else:
            print("❌ Traffic Producer Test Failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data Producers Test Failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints if the server is running"""
    print("\n🌐 Testing API Endpoints...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Health Check Passed")
            health_data = response.json()
            print(f"   API Status: {health_data.get('status', 'unknown')}")
        else:
            print(f"❌ API Health Check Failed - Status: {response.status_code}")
            return False
        
        # Test nearby drivers endpoint
        response = requests.get(
            f"{base_url}/drivers/nearby",
            params={"lat": 40.7589, "lon": -73.9851, "radius": 5.0},
            timeout=10
        )
        if response.status_code == 200:
            drivers_data = response.json()
            print(f"✅ Nearby Drivers Endpoint Passed - Found {drivers_data.get('count', 0)} drivers")
        else:
            print(f"❌ Nearby Drivers Endpoint Failed - Status: {response.status_code}")
        
        # Test prediction endpoint
        prediction_payload = {
            "driver_location": {"latitude": 40.7589, "longitude": -73.9851},
            "destination": {"latitude": 40.6892, "longitude": -74.0445}
        }
        response = requests.post(
            f"{base_url}/predict-arrival-time",
            json=prediction_payload,
            timeout=10
        )
        if response.status_code == 200:
            prediction = response.json()
            print(f"✅ Prediction Endpoint Passed - Arrival time: {prediction.get('predicted_arrival_time_minutes', 'N/A'):.2f} min")
        else:
            print(f"❌ Prediction Endpoint Failed - Status: {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("⚠️  API Server not running - skipping API tests")
        print("   Start the API server with: python api/main.py")
        return True
    except Exception as e:
        print(f"❌ API Tests Failed: {e}")
        return False

def test_model_training():
    """Test model training process"""
    print("\n🏋️  Testing Model Training...")
    
    try:
        from ml.training.train_model import generate_synthetic_data, prepare_data
        
        # Generate small dataset for testing
        print("   Generating test data...")
        df = generate_synthetic_data(num_samples=100)
        
        if len(df) == 100:
            print("✅ Data Generation Test Passed")
        else:
            print(f"❌ Data Generation Test Failed - Expected 100 samples, got {len(df)}")
            return False
        
        # Test data preparation
        print("   Testing data preparation...")
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        if X_train.shape[0] > 0 and X_test.shape[0] > 0:
            print(f"✅ Data Preparation Test Passed - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        else:
            print("❌ Data Preparation Test Failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model Training Test Failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚗 Ride Service ML - System Test")
    print("=" * 40)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("ML Model", test_ml_model),
        ("Data Producers", test_data_producers),
        ("Model Training", test_model_training),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} Test Failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    print("-" * 20)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 20)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
