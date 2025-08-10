from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
import redis
import json
import uuid
from datetime import datetime
import os
import sys

# Add the parent directory to the path to import ML modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.inference.model_service import get_model_service
from data.producers.gps_producer import GPSProducer
from data.producers.traffic_producer import TrafficProducer

app = FastAPI(
    title="Ride Service API",
    description="API for ride-sharing service with ML-powered arrival time predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_service = None
gps_producer = None
traffic_producer = None
redis_client = None

# Pydantic models
class Location(BaseModel):
    latitude: float
    longitude: float

class RideRequest(BaseModel):
    pickup_location: Location
    destination: Location
    user_id: str

class Driver(BaseModel):
    driver_id: str
    location: Location
    status: str
    distance_km: Optional[float] = None

class RideResponse(BaseModel):
    ride_id: str
    driver: Driver
    pickup_location: Location
    destination: Location
    predicted_arrival_time_minutes: float
    estimated_ride_duration_minutes: float
    status: str
    created_at: str

class PredictionRequest(BaseModel):
    driver_location: Location
    destination: Location
    traffic_factor: Optional[float] = 0.5

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model_service, gps_producer, traffic_producer, redis_client
    
    print("Starting up Ride Service API...")
    
    # Initialize ML model service
    model_service = get_model_service()
    
    # Initialize data producers
    gps_producer = GPSProducer()
    traffic_producer = TrafficProducer()
    
    # Initialize Redis
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        print("Connected to Redis")
    except Exception as e:
        print(f"Could not connect to Redis: {e}")
        redis_client = None
    
    print("Ride Service API started successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ride Service API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "ml_model": model_service is not None,
            "redis": redis_client is not None,
            "gps_producer": gps_producer is not None,
            "traffic_producer": traffic_producer is not None
        }
    }
    
    if model_service:
        ml_health = model_service.health_check()
        health_status["ml_service"] = ml_health
    
    return health_status

@app.post("/predict-arrival-time")
async def predict_arrival_time(request: PredictionRequest):
    """Predict arrival time for a ride"""
    try:
        if not model_service:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        # Get traffic factor for the route
        route_traffic = traffic_producer.get_route_traffic(
            request.driver_location.latitude,
            request.driver_location.longitude,
            request.destination.latitude,
            request.destination.longitude
        )
        
        # Use route traffic factor if not provided
        traffic_factor = request.traffic_factor or route_traffic['route_traffic_factor']
        
        # Make prediction
        prediction = model_service.predict_arrival_time(
            driver_location=(request.driver_location.latitude, request.driver_location.longitude),
            destination=(request.destination.latitude, request.destination.longitude),
            traffic_factor=traffic_factor
        )
        
        # Add route traffic information
        prediction['route_traffic_info'] = route_traffic
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/drivers/nearby")
async def get_nearby_drivers(lat: float, lon: float, radius: float = 5.0):
    """Get available drivers near a location"""
    try:
        if not gps_producer:
            raise HTTPException(status_code=503, detail="GPS service not available")
        
        nearby_drivers = gps_producer.get_available_drivers(
            location=(lat, lon),
            radius_km=radius
        )
        
        drivers = [
            Driver(
                driver_id=driver['driver_id'],
                location=Location(
                    latitude=driver['latitude'],
                    longitude=driver['longitude']
                ),
                status="available",
                distance_km=driver['distance_km']
            )
            for driver in nearby_drivers
        ]
        
        return {
            "drivers": drivers,
            "count": len(drivers),
            "search_radius_km": radius,
            "search_location": {"latitude": lat, "longitude": lon}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get nearby drivers: {str(e)}")

@app.post("/rides/request")
async def request_ride(ride_request: RideRequest):
    """Request a ride"""
    try:
        # Find nearby drivers
        nearby_drivers = gps_producer.get_available_drivers(
            location=(ride_request.pickup_location.latitude, ride_request.pickup_location.longitude),
            radius_km=10.0
        )
        
        if not nearby_drivers:
            raise HTTPException(status_code=404, detail="No available drivers found")
        
        # Select the closest driver
        selected_driver = nearby_drivers[0]
        
        # Get traffic conditions for the route
        route_traffic = traffic_producer.get_route_traffic(
            selected_driver['latitude'],
            selected_driver['longitude'],
            ride_request.pickup_location.latitude,
            ride_request.pickup_location.longitude
        )
        
        # Predict arrival time to pickup location
        pickup_prediction = model_service.predict_arrival_time(
            driver_location=(selected_driver['latitude'], selected_driver['longitude']),
            destination=(ride_request.pickup_location.latitude, ride_request.pickup_location.longitude),
            traffic_factor=route_traffic['route_traffic_factor']
        )
        
        # Predict ride duration
        ride_traffic = traffic_producer.get_route_traffic(
            ride_request.pickup_location.latitude,
            ride_request.pickup_location.longitude,
            ride_request.destination.latitude,
            ride_request.destination.longitude
        )
        
        ride_duration_prediction = model_service.predict_arrival_time(
            driver_location=(ride_request.pickup_location.latitude, ride_request.pickup_location.longitude),
            destination=(ride_request.destination.latitude, ride_request.destination.longitude),
            traffic_factor=ride_traffic['route_traffic_factor']
        )
        
        # Create ride record
        ride_id = str(uuid.uuid4())
        ride_data = {
            "ride_id": ride_id,
            "user_id": ride_request.user_id,
            "driver_id": selected_driver['driver_id'],
            "pickup_location": {
                "latitude": ride_request.pickup_location.latitude,
                "longitude": ride_request.pickup_location.longitude
            },
            "destination": {
                "latitude": ride_request.destination.latitude,
                "longitude": ride_request.destination.longitude
            },
            "predicted_arrival_time_minutes": pickup_prediction['predicted_arrival_time_minutes'],
            "estimated_ride_duration_minutes": ride_duration_prediction['predicted_arrival_time_minutes'],
            "status": "requested",
            "created_at": datetime.now().isoformat(),
            "driver_location": {
                "latitude": selected_driver['latitude'],
                "longitude": selected_driver['longitude']
            }
        }
        
        # Store in Redis if available
        if redis_client:
            redis_client.setex(f"ride:{ride_id}", 3600, json.dumps(ride_data))
        
        # Create response
        response = RideResponse(
            ride_id=ride_id,
            driver=Driver(
                driver_id=selected_driver['driver_id'],
                location=Location(
                    latitude=selected_driver['latitude'],
                    longitude=selected_driver['longitude']
                ),
                status="assigned",
                distance_km=selected_driver['distance_km']
            ),
            pickup_location=ride_request.pickup_location,
            destination=ride_request.destination,
            predicted_arrival_time_minutes=pickup_prediction['predicted_arrival_time_minutes'],
            estimated_ride_duration_minutes=ride_duration_prediction['predicted_arrival_time_minutes'],
            status="confirmed",
            created_at=ride_data['created_at']
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to request ride: {str(e)}")

@app.get("/rides/{ride_id}")
async def get_ride_status(ride_id: str):
    """Get ride status"""
    try:
        if not redis_client:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        ride_data = redis_client.get(f"ride:{ride_id}")
        if not ride_data:
            raise HTTPException(status_code=404, detail="Ride not found")
        
        ride_info = json.loads(ride_data)
        return ride_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ride status: {str(e)}")

@app.get("/traffic/location")
async def get_traffic_for_location(lat: float, lon: float):
    """Get traffic conditions for a specific location"""
    try:
        if not traffic_producer:
            raise HTTPException(status_code=503, detail="Traffic service not available")
        
        traffic_info = traffic_producer.get_traffic_for_location(lat, lon)
        return traffic_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get traffic info: {str(e)}")

@app.get("/traffic/route")
async def get_route_traffic(start_lat: float, start_lon: float, end_lat: float, end_lon: float):
    """Get traffic conditions along a route"""
    try:
        if not traffic_producer:
            raise HTTPException(status_code=503, detail="Traffic service not available")
        
        route_traffic = traffic_producer.get_route_traffic(start_lat, start_lon, end_lat, end_lon)
        return route_traffic
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get route traffic: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
