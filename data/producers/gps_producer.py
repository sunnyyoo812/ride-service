import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from typing import Dict, List, Tuple
import threading

class GPSProducer:
    """Producer for simulating GPS location data from drivers"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.running = False
        
        # NYC bounds for simulation
        self.nyc_bounds = {
            'lat_min': 40.4774,
            'lat_max': 40.9176,
            'lon_min': -74.2591,
            'lon_max': -73.7004
        }
        
        # Simulate 50 active drivers
        self.drivers = self._initialize_drivers(50)
        
        self._connect_kafka()
    
    def _connect_kafka(self):
        """Connect to Kafka"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            print("Connected to Kafka")
        except Exception as e:
            print(f"Failed to connect to Kafka: {e}")
    
    def _initialize_drivers(self, num_drivers: int) -> List[Dict]:
        """Initialize simulated drivers with random locations"""
        drivers = []
        
        for i in range(num_drivers):
            driver = {
                'driver_id': f'driver_{i:03d}',
                'lat': random.uniform(self.nyc_bounds['lat_min'], self.nyc_bounds['lat_max']),
                'lon': random.uniform(self.nyc_bounds['lon_min'], self.nyc_bounds['lon_max']),
                'status': random.choice(['available', 'busy', 'offline']),
                'speed': random.uniform(0, 60),  # km/h
                'heading': random.uniform(0, 360),  # degrees
                'last_update': datetime.now()
            }
            drivers.append(driver)
        
        return drivers
    
    def _update_driver_location(self, driver: Dict) -> Dict:
        """Update driver location with realistic movement"""
        # Simulate movement based on speed and heading
        speed_ms = driver['speed'] / 3.6  # Convert km/h to m/s
        time_delta = 5  # 5 seconds between updates
        
        # Simple movement calculation (not accounting for earth curvature)
        distance = speed_ms * time_delta  # meters
        
        # Convert to lat/lon changes (approximate)
        lat_change = (distance * 0.000009) * random.uniform(-1, 1)
        lon_change = (distance * 0.000009) * random.uniform(-1, 1)
        
        new_lat = driver['lat'] + lat_change
        new_lon = driver['lon'] + lon_change
        
        # Keep within NYC bounds
        new_lat = max(self.nyc_bounds['lat_min'], min(self.nyc_bounds['lat_max'], new_lat))
        new_lon = max(self.nyc_bounds['lon_min'], min(self.nyc_bounds['lon_max'], new_lon))
        
        # Occasionally change status, speed, and heading
        if random.random() < 0.1:  # 10% chance
            driver['status'] = random.choice(['available', 'busy', 'offline'])
        
        if random.random() < 0.2:  # 20% chance
            driver['speed'] = max(0, driver['speed'] + random.uniform(-10, 10))
        
        if random.random() < 0.15:  # 15% chance
            driver['heading'] = (driver['heading'] + random.uniform(-45, 45)) % 360
        
        driver.update({
            'lat': new_lat,
            'lon': new_lon,
            'last_update': datetime.now()
        })
        
        return driver
    
    def produce_gps_data(self):
        """Produce GPS data for all drivers"""
        if not self.producer:
            print("Kafka producer not connected")
            return
        
        for driver in self.drivers:
            # Update driver location
            updated_driver = self._update_driver_location(driver)
            
            # Create GPS message
            gps_message = {
                'driver_id': updated_driver['driver_id'],
                'latitude': updated_driver['lat'],
                'longitude': updated_driver['lon'],
                'status': updated_driver['status'],
                'speed': updated_driver['speed'],
                'heading': updated_driver['heading'],
                'timestamp': updated_driver['last_update'].isoformat(),
                'message_type': 'gps_update'
            }
            
            try:
                # Send to Kafka topic
                self.producer.send(
                    'driver_locations',
                    key=updated_driver['driver_id'],
                    value=gps_message
                )
            except Exception as e:
                print(f"Error sending GPS data: {e}")
        
        # Flush to ensure messages are sent
        self.producer.flush()
    
    def start_streaming(self, interval: int = 5):
        """Start streaming GPS data at regular intervals"""
        self.running = True
        print(f"Starting GPS data streaming every {interval} seconds...")
        
        while self.running:
            try:
                self.produce_gps_data()
                print(f"Sent GPS updates for {len(self.drivers)} drivers")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("Stopping GPS producer...")
                break
            except Exception as e:
                print(f"Error in GPS streaming: {e}")
                time.sleep(interval)
    
    def stop_streaming(self):
        """Stop streaming GPS data"""
        self.running = False
        if self.producer:
            self.producer.close()
    
    def get_available_drivers(self, location: Tuple[float, float], radius_km: float = 5) -> List[Dict]:
        """Get available drivers within radius of a location"""
        available_drivers = []
        target_lat, target_lon = location
        
        for driver in self.drivers:
            if driver['status'] == 'available':
                # Calculate distance (simplified)
                lat_diff = abs(driver['lat'] - target_lat)
                lon_diff = abs(driver['lon'] - target_lon)
                distance_approx = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111  # Rough km conversion
                
                if distance_approx <= radius_km:
                    available_drivers.append({
                        'driver_id': driver['driver_id'],
                        'latitude': driver['lat'],
                        'longitude': driver['lon'],
                        'distance_km': round(distance_approx, 2)
                    })
        
        return sorted(available_drivers, key=lambda x: x['distance_km'])

def main():
    """Main function to run GPS producer"""
    producer = GPSProducer()
    
    try:
        producer.start_streaming(interval=5)
    except KeyboardInterrupt:
        print("Shutting down GPS producer...")
    finally:
        producer.stop_streaming()

if __name__ == "__main__":
    main()
