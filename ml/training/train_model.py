import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.arrival_time_model import ArrivalTimePredictor, extract_features

class RideDataset(Dataset):
    """Dataset class for ride data"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def generate_synthetic_data(num_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic ride data for training
    """
    np.random.seed(42)
    
    # NYC approximate bounds
    lat_min, lat_max = 40.4774, 40.9176
    lon_min, lon_max = -74.2591, -73.7004
    
    data = []
    
    for _ in range(num_samples):
        # Random driver and destination locations in NYC
        driver_lat = np.random.uniform(lat_min, lat_max)
        driver_lon = np.random.uniform(lon_min, lon_max)
        dest_lat = np.random.uniform(lat_min, lat_max)
        dest_lon = np.random.uniform(lon_min, lon_max)
        
        # Random time features
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        
        # Traffic factor based on time of day
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            traffic_factor = np.random.uniform(0.7, 1.0)
        elif 22 <= hour or hour <= 6:  # Night hours
            traffic_factor = np.random.uniform(0.1, 0.4)
        else:  # Regular hours
            traffic_factor = np.random.uniform(0.3, 0.7)
        
        # Extract features
        features = extract_features(
            (driver_lat, driver_lon),
            (dest_lat, dest_lon),
            traffic_factor,
            hour,
            day_of_week
        )
        
        # Calculate actual arrival time with some noise
        base_time = features['historical_avg']
        traffic_multiplier = 1 + (traffic_factor * 0.5)  # Traffic can increase time by up to 50%
        noise = np.random.normal(0, 2)  # Add some random noise
        actual_time = max(1.0, base_time * traffic_multiplier + noise)
        
        data.append({
            'driver_lat': driver_lat,
            'driver_lon': driver_lon,
            'dest_lat': dest_lat,
            'dest_lon': dest_lon,
            'traffic_factor': traffic_factor,
            'hour': hour,
            'day_of_week': day_of_week,
            'historical_avg': features['historical_avg'],
            'actual_time': actual_time
        })
    
    return pd.DataFrame(data)

def prepare_data(df: pd.DataFrame):
    """Prepare data for training"""
    
    # Features
    feature_columns = ['driver_lat', 'driver_lon', 'dest_lat', 'dest_lon', 
                      'traffic_factor', 'hour', 'day_of_week', 'historical_avg']
    X = df[feature_columns].values
    y = df['actual_time'].values.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the neural network model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluate the trained model"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            outputs = model(features)
            predictions.extend(outputs.numpy().flatten())
            actuals.extend(targets.numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    print(f"Test Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return mse, rmse, mae

def main():
    """Main training function"""
    print("Generating synthetic training data...")
    df = generate_synthetic_data(num_samples=10000)
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Create datasets and data loaders
    train_dataset = RideDataset(X_train, y_train)
    test_dataset = RideDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = ArrivalTimePredictor(input_size=8, hidden_sizes=[64, 32, 16])
    
    print("Training model...")
    train_losses, val_losses = train_model(model, train_loader, test_loader)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    print("Evaluating model...")
    evaluate_model(model, test_loader)
    
    # Save model and scaler
    model_dir = '../inference'
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), f'{model_dir}/arrival_time_model.pth')
    
    with open(f'{model_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully!")
    
    # Clean up temporary file
    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')

if __name__ == "__main__":
    main()
