import streamlit as st
import requests
import json
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Ride Service",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def get_nearby_drivers(lat, lon, radius=5.0):
    """Get nearby drivers from the API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/drivers/nearby",
            params={"lat": lat, "lon": lon, "radius": radius},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_arrival_time(driver_lat, driver_lon, dest_lat, dest_lon):
    """Get arrival time prediction"""
    try:
        payload = {
            "driver_location": {"latitude": driver_lat, "longitude": driver_lon},
            "destination": {"latitude": dest_lat, "longitude": dest_lon}
        }
        response = requests.post(
            f"{API_BASE_URL}/predict-arrival-time",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def request_ride(pickup_lat, pickup_lon, dest_lat, dest_lon, user_id):
    """Request a ride"""
    try:
        payload = {
            "pickup_location": {"latitude": pickup_lat, "longitude": pickup_lon},
            "destination": {"latitude": dest_lat, "longitude": dest_lon},
            "user_id": user_id
        }
        response = requests.post(
            f"{API_BASE_URL}/rides/request",
            json=payload,
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_traffic_info(lat, lon):
    """Get traffic information for a location"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/traffic/location",
            params={"lat": lat, "lon": lon},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    st.markdown('<h1 class="main-header">🚗 Ride Service with ML Predictions</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API service is not available. Please make sure the backend is running.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # API Health Status
    st.sidebar.subheader("System Status")
    if health_data:
        services = health_data.get('services', {})
        for service, status in services.items():
            if status == "running" or status is True:
                st.sidebar.markdown(f"✅ {service.replace('_', ' ').title()}")
            else:
                st.sidebar.markdown(f"❌ {service.replace('_', ' ').title()}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "🚗 Request Ride", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Live Driver Locations")
        
        # NYC center coordinates
        nyc_center = [40.7589, -73.9851]
        
        # Create map
        m = folium.Map(location=nyc_center, zoom_start=12)
        
        # Get nearby drivers
        drivers_data = get_nearby_drivers(nyc_center[0], nyc_center[1], radius=20)
        
        if drivers_data and drivers_data.get('drivers'):
            drivers = drivers_data['drivers']
            
            # Add driver markers
            for driver in drivers:
                folium.Marker(
                    location=[driver['location']['latitude'], driver['location']['longitude']],
                    popup=f"Driver: {driver['driver_id']}<br>Status: {driver['status']}<br>Distance: {driver.get('distance_km', 'N/A')} km",
                    icon=folium.Icon(color='green', icon='car', prefix='fa')
                ).add_to(m)
            
            st.write(f"📍 Showing {len(drivers)} available drivers")
        else:
            st.warning("No driver data available")
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
        
        # Traffic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Traffic Information")
            if st.button("Get Traffic for Times Square"):
                traffic_info = get_traffic_info(40.7589, -73.9851)
                if traffic_info:
                    st.json(traffic_info)
        
        with col2:
            st.subheader("Driver Statistics")
            if drivers_data:
                st.metric("Available Drivers", drivers_data.get('count', 0))
                st.metric("Search Radius", f"{drivers_data.get('search_radius_km', 0)} km")
    
    with tab2:
        st.subheader("Request a Ride")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Pickup Location**")
            pickup_lat = st.number_input("Pickup Latitude", value=40.7589, format="%.6f")
            pickup_lon = st.number_input("Pickup Longitude", value=-73.9851, format="%.6f")
            
            st.write("**Destination**")
            dest_lat = st.number_input("Destination Latitude", value=40.6892, format="%.6f")
            dest_lon = st.number_input("Destination Longitude", value=-74.0445, format="%.6f")
            
            user_id = st.text_input("User ID", value="user_123")
        
        with col2:
            st.write("**Prediction Preview**")
            if st.button("Get Arrival Time Prediction"):
                # Find nearest driver first
                drivers_data = get_nearby_drivers(pickup_lat, pickup_lon, radius=10)
                
                if drivers_data and drivers_data.get('drivers'):
                    nearest_driver = drivers_data['drivers'][0]
                    driver_lat = nearest_driver['location']['latitude']
                    driver_lon = nearest_driver['location']['longitude']
                    
                    prediction = predict_arrival_time(driver_lat, driver_lon, pickup_lat, pickup_lon)
                    
                    if prediction:
                        st.success(f"🕐 Estimated arrival: {prediction['predicted_arrival_time_minutes']:.1f} minutes")
                        st.info(f"🚗 Nearest driver: {nearest_driver['driver_id']} ({nearest_driver['distance_km']:.1f} km away)")
                        
                        if 'route_traffic_info' in prediction:
                            traffic = prediction['route_traffic_info']
                            st.write(f"🚦 Traffic: {traffic['route_congestion_level'].title()}")
                    else:
                        st.error("Failed to get prediction")
                else:
                    st.warning("No drivers available in the area")
        
        st.divider()
        
        # Request ride button
        if st.button("🚗 Request Ride", type="primary"):
            with st.spinner("Requesting ride..."):
                ride_response = request_ride(pickup_lat, pickup_lon, dest_lat, dest_lon, user_id)
                
                if ride_response:
                    st.success("✅ Ride confirmed!")
                    
                    # Display ride details
                    st.subheader("Ride Details")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Ride ID", ride_response['ride_id'][:8] + "...")
                        st.metric("Driver", ride_response['driver']['driver_id'])
                    
                    with col2:
                        st.metric("Arrival Time", f"{ride_response['predicted_arrival_time_minutes']:.1f} min")
                        st.metric("Ride Duration", f"{ride_response['estimated_ride_duration_minutes']:.1f} min")
                    
                    with col3:
                        st.metric("Status", ride_response['status'].title())
                        st.metric("Distance", f"{ride_response['driver']['distance_km']:.1f} km")
                    
                    # Store ride ID in session state
                    st.session_state.current_ride_id = ride_response['ride_id']
                    
                else:
                    st.error("❌ Failed to request ride. Please try again.")
    
    with tab3:
        st.subheader("Analytics Dashboard")
        
        # Mock analytics data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rides Today", "127", "12%")
        
        with col2:
            st.metric("Average Wait Time", "4.2 min", "-0.8 min")
        
        with col3:
            st.metric("Active Drivers", "45", "3")
        
        with col4:
            st.metric("Prediction Accuracy", "94.2%", "1.1%")
        
        # Charts
        st.subheader("Performance Metrics")
        
        # Mock data for charts
        chart_data = pd.DataFrame({
            'Hour': list(range(24)),
            'Rides': [5, 3, 2, 1, 1, 2, 8, 15, 12, 8, 10, 12, 15, 14, 16, 18, 22, 25, 20, 15, 12, 10, 8, 6],
            'Avg_Wait_Time': [3.2, 2.8, 2.5, 2.1, 2.0, 2.3, 4.1, 5.8, 5.2, 4.1, 4.5, 4.8, 5.1, 4.9, 5.3, 5.7, 6.2, 6.8, 6.1, 5.4, 4.8, 4.2, 3.8, 3.5]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(chart_data.set_index('Hour')['Rides'])
            st.caption("Rides per Hour")
        
        with col2:
            st.line_chart(chart_data.set_index('Hour')['Avg_Wait_Time'])
            st.caption("Average Wait Time (minutes)")
    
    with tab4:
        st.subheader("Settings")
        
        st.write("**API Configuration**")
        st.text_input("API Base URL", value=API_BASE_URL, disabled=True)
        
        st.write("**Map Settings**")
        default_radius = st.slider("Default Search Radius (km)", 1, 20, 5)
        
        st.write("**Refresh Settings**")
        auto_refresh = st.checkbox("Auto-refresh driver locations")
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 30)
        
        st.write("**System Information**")
        if health_data:
            st.json(health_data)

if __name__ == "__main__":
    main()
