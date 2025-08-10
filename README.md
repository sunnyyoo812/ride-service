# Ride Service ML - Car Ride Service with Machine Learning Predictions

A comprehensive ride-sharing service that uses machine learning to predict arrival times, featuring live streaming data, model orchestration, and containerized deployment.

## 🚀 Features

- **ML-Powered Predictions**: Neural network model predicting ride arrival times
- **Live Data Streaming**: Real-time GPS and traffic data via Kafka
- **Model Orchestration**: Automated training and deployment with Airflow
- **Containerized Architecture**: Full Docker deployment
- **Interactive Frontend**: Streamlit web interface with live maps
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Production Ready**: Redis caching, PostgreSQL storage, health checks

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │   ML Service    │
│   Frontend      │◄──►│    Backend      │◄──►│   (PyTorch)     │
│   (Port 8501)   │    │   (Port 8000)   │    │   (Port 8001)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌▼────────────────┐    ┌─────────────────┐
         │     Kafka       │    │     Redis       │    │   PostgreSQL    │
         │  (Streaming)    │    │   (Caching)     │    │   (Database)    │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
                 ▲
         ┌───────┴───────┐
         │               │
┌────────▼──────┐ ┌──────▼──────┐
│ GPS Producer  │ │Traffic      │
│ (Driver Data) │ │Producer     │
└───────────────┘ └─────────────┘
```

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- 8GB+ RAM recommended

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ride-service-ml
```

### 2. Start the System
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 3. Access the Applications

- **Frontend (Streamlit)**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

### 4. Train the ML Model (Optional)
```bash
# Train the model locally
cd ml/training
python train_model.py
```

## 🛠️ Development Setup

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start individual services
python api/main.py                    # API server
streamlit run frontend/streamlit_app.py  # Frontend
python data/producers/gps_producer.py    # GPS data
python data/producers/traffic_producer.py # Traffic data
```

### Training the Model
```bash
cd ml/training
python train_model.py
```

## 📊 API Endpoints

### Core Endpoints
- `GET /health` - System health check
- `POST /predict-arrival-time` - Get arrival time prediction
- `GET /drivers/nearby` - Find nearby drivers
- `POST /rides/request` - Request a ride
- `GET /rides/{ride_id}` - Get ride status

### Data Endpoints
- `GET /traffic/location` - Traffic for specific location
- `GET /traffic/route` - Traffic along a route

### Example API Usage
```python
import requests

# Get nearby drivers
response = requests.get(
    "http://localhost:8000/drivers/nearby",
    params={"lat": 40.7589, "lon": -73.9851, "radius": 5.0}
)

# Predict arrival time
response = requests.post(
    "http://localhost:8000/predict-arrival-time",
    json={
        "driver_location": {"latitude": 40.7589, "longitude": -73.9851},
        "destination": {"latitude": 40.6892, "longitude": -74.0445}
    }
)

# Request a ride
response = requests.post(
    "http://localhost:8000/rides/request",
    json={
        "pickup_location": {"latitude": 40.7589, "longitude": -73.9851},
        "destination": {"latitude": 40.6892, "longitude": -74.0445},
        "user_id": "user_123"
    }
)
```

## 🧠 Machine Learning Model

### Model Architecture
- **Type**: Feed-forward Neural Network (PyTorch)
- **Input Features**: 8 features including GPS coordinates, traffic conditions, time features
- **Architecture**: 64 → 32 → 16 → 1 neurons with ReLU activation and dropout
- **Output**: Predicted arrival time in minutes

### Features Used
1. Driver GPS coordinates (lat, lon)
2. Destination coordinates (lat, lon)
3. Traffic factor (0-1 scale)
4. Hour of day (0-23)
5. Day of week (0-6)
6. Historical average time

### Training Process
1. **Data Generation**: Synthetic NYC ride data with realistic patterns
2. **Feature Engineering**: Distance calculation, time-based features
3. **Model Training**: Adam optimizer with learning rate scheduling
4. **Validation**: Early stopping and model checkpointing
5. **Deployment**: Automated via Airflow pipeline

## 📡 Data Streaming

### Kafka Topics
- `driver_locations`: Real-time GPS updates from drivers
- `traffic_conditions`: Traffic data from monitoring zones

### Data Producers
- **GPS Producer**: Simulates 50 drivers moving around NYC
- **Traffic Producer**: Generates traffic data for 100 zones

## 🔄 Model Orchestration (Airflow)

### Daily Pipeline
1. **Data Generation**: Create synthetic training data
2. **Model Training**: Train neural network
3. **Model Validation**: Test predictions and performance
4. **Model Deployment**: Deploy validated model
5. **Cleanup**: Remove old model backups
6. **Health Check**: Verify system status

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| frontend | 8501 | Streamlit web interface |
| api-service | 8000 | FastAPI backend |
| ml-service | 8001 | ML model service |
| data-producer | - | GPS and traffic data generators |
| kafka | 9092 | Message streaming |
| zookeeper | 2181 | Kafka coordination |
| redis | 6379 | Caching layer |
| postgres | 5432 | Database |

## 🔧 Configuration

### Environment Variables
```bash
# API Service
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ride_service
REDIS_URL=redis://redis:6379
ML_SERVICE_URL=http://ml-service:8001

# ML Service
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
REDIS_URL=redis://redis:6379

# Data Producers
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
```

## 📈 Monitoring and Health Checks

### Health Endpoints
- API: `http://localhost:8000/health`
- Frontend: `http://localhost:8501/_stcore/health`

### Metrics Available
- System status for all services
- ML model performance metrics
- Real-time driver and traffic statistics
- Prediction accuracy tracking

## 🚨 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

2. **Kafka connection issues**
   ```bash
   # Wait for Kafka to be ready
   docker-compose logs kafka
   ```

3. **Model not found errors**
   ```bash
   # Train the model first
   docker-compose exec ml-service python ml/training/train_model.py
   ```

4. **Frontend not loading**
   ```bash
   # Check API connectivity
   curl http://localhost:8000/health
   ```

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api-service
docker-compose logs frontend
docker-compose logs ml-service
```

## 🔮 Future Enhancements

- [ ] Real traffic API integration (Google Maps, HERE)
- [ ] Mobile app development
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Real-time model updates
- [ ] A/B testing framework
- [ ] Kubernetes deployment
- [ ] Monitoring with Prometheus/Grafana
- [ ] Load testing and performance optimization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.
