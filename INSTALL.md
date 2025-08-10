# Installation Guide - Ride Service ML

This guide will help you install and run the Ride Service ML system.

## Prerequisites

### Required Software
1. **Docker Desktop** (recommended)
   - Download from: https://www.docker.com/products/docker-desktop/
   - Includes Docker and Docker Compose

2. **Alternative: Docker + Docker Compose separately**
   - Docker Engine: https://docs.docker.com/engine/install/
   - Docker Compose: https://docs.docker.com/compose/install/

3. **Python 3.9+** (for local development only)
   - Download from: https://www.python.org/downloads/

### System Requirements
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

## Quick Installation

### Option 1: Docker Desktop (Recommended)

1. **Install Docker Desktop**
   ```bash
   # Download and install Docker Desktop from the official website
   # Start Docker Desktop application
   ```

2. **Clone and Run**
   ```bash
   git clone <repository-url>
   cd ride-service-ml
   chmod +x start.sh
   ./start.sh
   ```

3. **Access Applications**
   - Frontend: http://localhost:8501
   - API: http://localhost:8000/docs

### Option 2: Manual Docker Installation

1. **Install Docker Engine**
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # macOS (using Homebrew)
   brew install docker
   
   # Windows: Download Docker Desktop
   ```

2. **Install Docker Compose**
   ```bash
   # Linux
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   
   # macOS (using Homebrew)
   brew install docker-compose
   
   # Windows: Included with Docker Desktop
   ```

3. **Run the System**
   ```bash
   git clone <repository-url>
   cd ride-service-ml
   docker-compose up -d
   ```

## Local Development Setup

If you want to run components locally for development:

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Individual Services**
   ```bash
   # API Server
   python api/main.py
   
   # Frontend
   streamlit run frontend/streamlit_app.py
   
   # Data Producers
   python data/producers/gps_producer.py
   python data/producers/traffic_producer.py
   ```

## Troubleshooting

### Common Issues

1. **Docker not found**
   ```bash
   # Check if Docker is installed and running
   docker --version
   docker-compose --version
   ```

2. **Permission denied**
   ```bash
   # Linux: Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **Port conflicts**
   ```bash
   # Check what's using the ports
   lsof -i :8000  # API port
   lsof -i :8501  # Frontend port
   lsof -i :9092  # Kafka port
   ```

4. **Memory issues**
   ```bash
   # Increase Docker memory limit in Docker Desktop settings
   # Recommended: 6GB+ for all services
   ```

### Verification

1. **Check Services**
   ```bash
   docker-compose ps
   ```

2. **View Logs**
   ```bash
   docker-compose logs
   ```

3. **Test API**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Run Tests**
   ```bash
   python test_system.py
   ```

## Next Steps

Once installed:

1. **Access the Frontend**: http://localhost:8501
2. **Explore the API**: http://localhost:8000/docs
3. **Check System Health**: http://localhost:8000/health
4. **View the README**: See README.md for detailed usage

## Getting Help

If you encounter issues:

1. Check the logs: `docker-compose logs`
2. Restart services: `./start.sh restart`
3. Clean restart: `./start.sh cleanup && ./start.sh`
4. Check system requirements above
5. Open an issue on GitHub

## Uninstallation

To completely remove the system:

```bash
# Stop and remove containers
docker-compose down -v

# Remove images (optional)
docker rmi $(docker images "ride-service-ml*" -q)

# Clean up Docker system (optional)
docker system prune -a
