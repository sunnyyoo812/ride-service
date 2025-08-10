import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class ArrivalTimePredictor(nn.Module):
    """
    Neural network model to predict ride arrival times based on:
    - Driver GPS coordinates (lat, lon)
    - Destination coordinates (lat, lon)
    - Current traffic conditions (0-1 scale)
    - Time features (hour, day_of_week)
    - Historical patterns
    """
    
    def __init__(self, input_size: int = 8, hidden_sizes: List[int] = [64, 32, 16], output_size: int = 1):
        super(ArrivalTimePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
               Features: [driver_lat, driver_lon, dest_lat, dest_lon, 
                         traffic_factor, hour, day_of_week, historical_avg]
        
        Returns:
            Predicted arrival time in minutes
        """
        return self.network(x)
    
    def predict_arrival_time(self, features: Dict) -> float:
        """
        Predict arrival time for a single ride request
        
        Args:
            features: Dictionary containing:
                - driver_lat: Driver latitude
                - driver_lon: Driver longitude
                - dest_lat: Destination latitude
                - dest_lon: Destination longitude
                - traffic_factor: Traffic condition (0-1)
                - hour: Hour of day (0-23)
                - day_of_week: Day of week (0-6)
                - historical_avg: Historical average time for similar routes
        
        Returns:
            Predicted arrival time in minutes
        """
        self.eval()
        
        # Convert features to tensor
        feature_vector = torch.tensor([
            features['driver_lat'],
            features['driver_lon'],
            features['dest_lat'],
            features['dest_lon'],
            features['traffic_factor'],
            features['hour'],
            features['day_of_week'],
            features['historical_avg']
        ], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.forward(feature_vector)
            return max(1.0, prediction.item())  # Minimum 1 minute arrival time

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two points in kilometers
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

def extract_features(driver_location: Tuple[float, float], 
                    destination: Tuple[float, float],
                    traffic_factor: float,
                    hour: int,
                    day_of_week: int,
                    historical_data: Dict = None) -> Dict:
    """
    Extract features for the ML model
    
    Args:
        driver_location: (lat, lon) of driver
        destination: (lat, lon) of destination
        traffic_factor: Current traffic condition (0-1)
        hour: Hour of day (0-23)
        day_of_week: Day of week (0-6, Monday=0)
        historical_data: Historical ride data for similar routes
    
    Returns:
        Dictionary of features for the model
    """
    driver_lat, driver_lon = driver_location
    dest_lat, dest_lon = destination
    
    # Calculate base distance
    distance = calculate_distance(driver_lat, driver_lon, dest_lat, dest_lon)
    
    # Estimate historical average based on distance and time of day
    # This is a simplified calculation - in production, use actual historical data
    base_speed = 30  # km/h base speed
    if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
        base_speed *= 0.6
    elif 22 <= hour or hour <= 6:  # Night hours
        base_speed *= 1.2
    
    historical_avg = (distance / base_speed) * 60  # Convert to minutes
    
    return {
        'driver_lat': driver_lat,
        'driver_lon': driver_lon,
        'dest_lat': dest_lat,
        'dest_lon': dest_lon,
        'traffic_factor': traffic_factor,
        'hour': hour,
        'day_of_week': day_of_week,
        'historical_avg': historical_avg
    }
