import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from typing import Dict, List
import math

class TrafficProducer:
    """Producer for simulating traffic condition data"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.running = False
        
        # NYC traffic zones (simplified grid)
        self.traffic_zones = self._initialize_traffic_zones()
        
        self._connect_kafka()
    
    def _connect_kafka(self):
        """Connect to Kafka"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            print("Connected to Kafka for traffic data")
        except Exception as e:
            print(f"Failed to connect to Kafka: {e}")
    
    def _initialize_traffic_zones(self) -> List[Dict]:
        """Initialize traffic monitoring zones across NYC"""
        zones = []
        
        # NYC bounds
        lat_min, lat_max = 40.4774, 40.9176
        lon_min, lon_max = -74.2591, -73.7004
        
        # Create a 10x10 grid of traffic zones
        lat_step = (lat_max - lat_min) / 10
        lon_step = (lon_max - lon_min) / 10
        
        zone_id = 0
        for i in range(10):
            for j in range(10):
                zone = {
                    'zone_id': f'zone_{zone_id:03d}',
                    'center_lat': lat_min + (i + 0.5) * lat_step,
                    'center_lon': lon_min + (j + 0.5) * lon_step,
                    'bounds': {
                        'lat_min': lat_min + i * lat_step,
                        'lat_max': lat_min + (i + 1) * lat_step,
                        'lon_min': lon_min + j * lon_step,
                        'lon_max': lon_min + (j + 1) * lon_step
                    },
                    'traffic_factor': random.uniform(0.2, 0.8),
                    'avg_speed': random.uniform(20, 50),  # km/h
                    'congestion_level': 'moderate',
                    'last_update': datetime.now()
                }
                zones.append(zone)
                zone_id += 1
        
        return zones
    
    def _calculate_time_based_traffic(self, base_factor: float, hour: int) -> float:
        """Calculate traffic factor based on time of day"""
        # Rush hour patterns
        if 7 <= hour <= 9:  # Morning rush
            multiplier = 1.5
        elif 17 <= hour <= 19:  # Evening rush
            multiplier = 1.6
        elif 12 <= hour <= 14:  # Lunch time
            multiplier = 1.2
        elif 22 <= hour or hour <= 6:  # Night time
            multiplier = 0.4
        else:  # Regular hours
            multiplier = 1.0
        
        # Add some randomness
        multiplier *= random.uniform(0.8, 1.2)
        
        # Calculate final traffic factor (0-1 scale, where 1 is worst traffic)
        traffic_factor = min(1.0, base_factor * multiplier)
        
        return traffic_factor
    
    def _update_zone_traffic(self, zone: Dict) -> Dict:
        """Update traffic conditions for a zone"""
        current_hour = datetime.now().hour
        
        # Calculate new traffic factor
        base_factor = zone['traffic_factor']
        new_traffic_factor = self._calculate_time_based_traffic(base_factor, current_hour)
        
        # Update average speed based on traffic
        base_speed = 40  # Base speed in km/h
        speed_reduction = new_traffic_factor * 0.7  # Traffic can reduce speed by up to 70%
        new_avg_speed = base_speed * (1 - speed_reduction)
        
        # Determine congestion level
        if new_traffic_factor < 0.3:
            congestion_level = 'light'
        elif new_traffic_factor < 0.6:
            congestion_level = 'moderate'
        elif new_traffic_factor < 0.8:
            congestion_level = 'heavy'
        else:
            congestion_level = 'severe'
        
        # Add some random variation
        new_traffic_factor += random.uniform(-0.05, 0.05)
        new_traffic_factor = max(0.0, min(1.0, new_traffic_factor))
        
        zone.update({
            'traffic_factor': new_traffic_factor,
            'avg_speed': max(5, new_avg_speed),  # Minimum 5 km/h
            'congestion_level': congestion_level,
            'last_update': datetime.now()
        })
        
        return zone
    
    def produce_traffic_data(self):
        """Produce traffic data for all zones"""
        if not self.producer:
            print("Kafka producer not connected")
            return
        
        for zone in self.traffic_zones:
            # Update zone traffic conditions
            updated_zone = self._update_zone_traffic(zone)
            
            # Create traffic message
            traffic_message = {
                'zone_id': updated_zone['zone_id'],
                'center_latitude': updated_zone['center_lat'],
                'center_longitude': updated_zone['center_lon'],
                'bounds': updated_zone['bounds'],
                'traffic_factor': round(updated_zone['traffic_factor'], 3),
                'average_speed_kmh': round(updated_zone['avg_speed'], 1),
                'congestion_level': updated_zone['congestion_level'],
                'timestamp': updated_zone['last_update'].isoformat(),
                'message_type': 'traffic_update'
            }
            
            try:
                # Send to Kafka topic
                self.producer.send(
                    'traffic_conditions',
                    key=updated_zone['zone_id'],
                    value=traffic_message
                )
            except Exception as e:
                print(f"Error sending traffic data: {e}")
        
        # Flush to ensure messages are sent
        self.producer.flush()
    
    def start_streaming(self, interval: int = 30):
        """Start streaming traffic data at regular intervals"""
        self.running = True
        print(f"Starting traffic data streaming every {interval} seconds...")
        
        while self.running:
            try:
                self.produce_traffic_data()
                print(f"Sent traffic updates for {len(self.traffic_zones)} zones")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("Stopping traffic producer...")
                break
            except Exception as e:
                print(f"Error in traffic streaming: {e}")
                time.sleep(interval)
    
    def stop_streaming(self):
        """Stop streaming traffic data"""
        self.running = False
        if self.producer:
            self.producer.close()
    
    def get_traffic_for_location(self, lat: float, lon: float) -> Dict:
        """Get traffic conditions for a specific location"""
        # Find the zone containing this location
        for zone in self.traffic_zones:
            bounds = zone['bounds']
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                bounds['lon_min'] <= lon <= bounds['lon_max']):
                return {
                    'zone_id': zone['zone_id'],
                    'traffic_factor': zone['traffic_factor'],
                    'average_speed': zone['avg_speed'],
                    'congestion_level': zone['congestion_level']
                }
        
        # If no zone found, return default values
        return {
            'zone_id': 'unknown',
            'traffic_factor': 0.5,
            'average_speed': 30,
            'congestion_level': 'moderate'
        }
    
    def get_route_traffic(self, start_lat: float, start_lon: float, 
                         end_lat: float, end_lon: float) -> Dict:
        """Get average traffic conditions along a route"""
        # Simple implementation: sample points along the route
        num_samples = 5
        traffic_factors = []
        speeds = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            sample_lat = start_lat + t * (end_lat - start_lat)
            sample_lon = start_lon + t * (end_lon - start_lon)
            
            traffic_info = self.get_traffic_for_location(sample_lat, sample_lon)
            traffic_factors.append(traffic_info['traffic_factor'])
            speeds.append(traffic_info['average_speed'])
        
        avg_traffic_factor = sum(traffic_factors) / len(traffic_factors)
        avg_speed = sum(speeds) / len(speeds)
        
        # Determine overall congestion level
        if avg_traffic_factor < 0.3:
            overall_congestion = 'light'
        elif avg_traffic_factor < 0.6:
            overall_congestion = 'moderate'
        elif avg_traffic_factor < 0.8:
            overall_congestion = 'heavy'
        else:
            overall_congestion = 'severe'
        
        return {
            'route_traffic_factor': round(avg_traffic_factor, 3),
            'route_average_speed': round(avg_speed, 1),
            'route_congestion_level': overall_congestion,
            'samples_analyzed': num_samples
        }

def main():
    """Main function to run traffic producer"""
    producer = TrafficProducer()
    
    try:
        producer.start_streaming(interval=30)
    except KeyboardInterrupt:
        print("Shutting down traffic producer...")
    finally:
        producer.stop_streaming()

if __name__ == "__main__":
    main()
