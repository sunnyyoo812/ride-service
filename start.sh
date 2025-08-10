#!/bin/bash

# Ride Service ML - Startup Script
# This script helps you get the ride service up and running quickly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Function to train the ML model
train_model() {
    print_status "Training ML model..."
    cd ml/training
    python train_model.py
    cd ../..
    print_success "Model training completed"
}

# Function to start services
start_services() {
    print_status "Starting all services with Docker Compose..."
    docker-compose up -d
    
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    print_status "Checking service health..."
    
    # Check API health
    for i in {1..10}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API service is healthy"
            break
        else
            print_warning "Waiting for API service... (attempt $i/10)"
            sleep 10
        fi
        
        if [ $i -eq 10 ]; then
            print_error "API service failed to start properly"
            docker-compose logs api-service
            exit 1
        fi
    done
    
    # Check Frontend
    for i in {1..5}; do
        if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
            print_success "Frontend service is healthy"
            break
        else
            print_warning "Waiting for frontend service... (attempt $i/5)"
            sleep 10
        fi
        
        if [ $i -eq 5 ]; then
            print_warning "Frontend service may not be fully ready yet"
        fi
    done
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_status "Service URLs:"
    echo "  🌐 Frontend (Streamlit): http://localhost:8501"
    echo "  🔧 API Documentation:   http://localhost:8000/docs"
    echo "  ❤️  API Health Check:    http://localhost:8000/health"
    echo ""
}

# Function to show logs
show_logs() {
    print_status "Recent logs from all services:"
    docker-compose logs --tail=50
}

# Function to stop services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "Ride Service ML - Startup Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services (default)"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show recent logs"
    echo "  train       Train the ML model locally"
    echo "  cleanup     Stop services and clean up Docker resources"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Start all services"
    echo "  $0 start        # Start all services"
    echo "  $0 status       # Check service status"
    echo "  $0 logs         # View logs"
    echo "  $0 stop         # Stop all services"
    echo ""
}

# Main script logic
main() {
    echo "🚗 Ride Service ML - Startup Script"
    echo "=================================="
    echo ""
    
    # Parse command line arguments
    COMMAND=${1:-start}
    
    case $COMMAND in
        "start")
            check_docker
            check_docker_compose
            start_services
            show_status
            print_success "🎉 Ride Service ML is now running!"
            print_status "You can now access the frontend at http://localhost:8501"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            check_docker
            check_docker_compose
            stop_services
            start_services
            show_status
            print_success "🎉 Ride Service ML has been restarted!"
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "train")
            if command -v python &> /dev/null; then
                train_model
            else
                print_error "Python is not available. Please install Python or use Docker to train the model."
                print_status "Alternative: docker-compose exec ml-service python ml/training/train_model.py"
            fi
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
