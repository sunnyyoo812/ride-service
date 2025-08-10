import torch
import pickle
import os
import redis
import json
from datetime import datetime
from typing import Dict, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.arrival_time_model import ArrivalTimePredictor, extract_features

class ModelService:
    """Service for loading and serving the trained arrival time prediction model"""
    
    def __init__(self, model_path: str = None, scaler_path: str = None, redis_url: str = None):
        self.model_path = model_path or 'arrival_time_model.pth'
        self.scaler_path = scaler_path or 'scaler.pkl'
        self.redis_url = redis_url or 'redis://localhost:6379'
        
        # Initialize model and scaler
        self.model = None
        self.scaler = None
        self.redis_client = None
        
        self._load_model()
        self._load_scaler()
        self._connect_redis()
    
    def _load_model(self):
        """Load the trained PyTorch model"""
        try:
            self.model = ArrivalTimePredictor(input_size=8, hidden_sizes=[64, 32, 16])
            
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                self.model.eval()
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"Model file {self.model_path} not found. Using untrained model.")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = ArrivalTimePredictor(input_size=8, hidden_sizes=[64, 32, 16])
    
    def _load_scaler(self):
        """Load the feature scaler"""
        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Scaler loaded from {self.scaler_path}")
            else:
                print(f"Scaler file {self.scaler_path} not found. Features will not be scaled.")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    
    def _connect_redis(self):
        """Connect to Redis for caching"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            print("Connected to Redis")
        except Exception as e:
            print(f"Could not connect to Redis: {e}")
            self.redis_client = None
    
    def predict_arrival_time(self, 
                           driver_location: Tuple[float, float],
                           destination: Tuple[float, float],
                           traffic_factor: float = 0.5) -> Dict:
        """
        Predict arrival time for a ride request
        
        Args:
            driver_location: (lat, lon) of driver
            destination: (lat, lon) of destination
            traffic_factor: Current traffic condition (0-1)
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get current time features
            now = datetime.now()
            hour = now.hour
            day_of_week = now.weekday()
            
            # Extract features
            features = extract_features(
                driver_location,
                destination,
                traffic_factor,
                hour,
                day_of_week
            )
            
            # Create feature vector
            feature_vector = [
                features['driver_lat'],
                features['driver_lon'],
                features['dest_lat'],
                features['dest_lon'],
                features['traffic_factor'],
                features['hour'],
                features['day_of_week'],
                features['historical_avg']
            ]
            
            # Scale features if scaler is available
            if self.scaler:
                import numpy as np
                feature_vector = self.scaler.transform([feature_vector])[0]
            
            # Make prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
                prediction = self.model(input_tensor)
                arrival_time = max(1.0, prediction.item())
            
            result = {
                'predicted_arrival_time_minutes': round(arrival_time, 2),
                'driver_location': driver_location,
                'destination': destination,
                'traffic_factor': traffic_factor,
                'timestamp': now.isoformat(),
                'features_used': features
            }
            
            # Cache result in Redis if available
            if self.redis_client:
                cache_key = f"prediction:{hash(str(driver_location + destination))}"
                self.redis_client.setex(cache_key, 300, json.dumps(result))  # Cache for 5 minutes
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {
                'error': str(e),
                'predicted_arrival_time_minutes': 10.0,  # Default fallback
                'timestamp': datetime.now().isoformat()
            }
    
    def get_cached_prediction(self, driver_location: Tuple[float, float], 
                            destination: Tuple[float, float]) -> Dict:
        """Get cached prediction if available"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"prediction:{hash(str(driver_location + destination))}"
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
        except Exception as e:
            print(f"Error retrieving cached prediction: {e}")
        
        return None
    
    def batch_predict(self, requests: list) -> list:
        """
        Make batch predictions for multiple ride requests
        
        Args:
            requests: List of dictionaries with 'driver_location', 'destination', 'traffic_factor'
        
        Returns:
            List of prediction results
        """
        results = []
        
        for request in requests:
            result = self.predict_arrival_time(
                request['driver_location'],
                request['destination'],
                request.get('traffic_factor', 0.5)
            )
            results.append(result)
        
        return results
    
    def health_check(self) -> Dict:
        """Health check for the model service"""
        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'redis_connected': self.redis_client is not None,
            'timestamp': datetime.now().isoformat()
        }

# Global model service instance
model_service = None

def get_model_service() -> ModelService:
    """Get or create the global model service instance"""
    global model_service
    if model_service is None:
        model_service = ModelService()
    return model_service

if __name__ == "__main__":
    # Test the model service
    service = ModelService()
    
    # Test prediction
    test_result = service.predict_arrival_time(
        driver_location=(40.7589, -73.9851),  # Times Square
        destination=(40.6892, -74.0445),      # Statue of Liberty
        traffic_factor=0.6
    )
    
    print("Test prediction result:")
    print(json.dumps(test_result, indent=2))
    
    # Health check
    health = service.health_check()
    print("\nHealth check:")
    print(json.dumps(health, indent=2))
