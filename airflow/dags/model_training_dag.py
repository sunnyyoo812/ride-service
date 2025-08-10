from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add the project root to Python path
sys.path.append('/app')

default_args = {
    'owner': 'ride-service',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='ML model training and deployment pipeline',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    tags=['ml', 'training', 'ride-service'],
)

def generate_training_data(**context):
    """Generate synthetic training data"""
    import sys
    sys.path.append('/app')
    
    from ml.training.train_model import generate_synthetic_data
    import pandas as pd
    
    print("Generating synthetic training data...")
    
    # Generate larger dataset for production
    df = generate_synthetic_data(num_samples=50000)
    
    # Save to temporary location
    data_path = '/tmp/training_data.csv'
    df.to_csv(data_path, index=False)
    
    print(f"Generated {len(df)} training samples and saved to {data_path}")
    
    # Store path in XCom for next task
    return data_path

def train_model(**context):
    """Train the ML model"""
    import sys
    sys.path.append('/app')
    
    from ml.training.train_model import main as train_main
    import os
    
    print("Starting model training...")
    
    # Change to the training directory
    original_dir = os.getcwd()
    os.chdir('/app/ml/training')
    
    try:
        # Run training
        train_main()
        print("Model training completed successfully!")
        return "training_completed"
    except Exception as e:
        print(f"Model training failed: {e}")
        raise
    finally:
        os.chdir(original_dir)

def validate_model(**context):
    """Validate the trained model"""
    import sys
    sys.path.append('/app')
    
    from ml.inference.model_service import ModelService
    import torch
    import os
    
    print("Validating trained model...")
    
    # Check if model files exist
    model_path = '/app/ml/inference/arrival_time_model.pth'
    scaler_path = '/app/ml/inference/scaler.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    # Test model loading and prediction
    try:
        service = ModelService(model_path=model_path, scaler_path=scaler_path)
        
        # Test prediction
        test_prediction = service.predict_arrival_time(
            driver_location=(40.7589, -73.9851),  # Times Square
            destination=(40.6892, -74.0445),      # Statue of Liberty
            traffic_factor=0.6
        )
        
        if 'predicted_arrival_time_minutes' not in test_prediction:
            raise ValueError("Model prediction missing required field")
        
        arrival_time = test_prediction['predicted_arrival_time_minutes']
        if arrival_time <= 0 or arrival_time > 120:  # Reasonable bounds
            raise ValueError(f"Model prediction out of reasonable bounds: {arrival_time}")
        
        print(f"Model validation successful! Test prediction: {arrival_time:.2f} minutes")
        return "validation_passed"
        
    except Exception as e:
        print(f"Model validation failed: {e}")
        raise

def deploy_model(**context):
    """Deploy the validated model"""
    import shutil
    import os
    from datetime import datetime
    
    print("Deploying validated model...")
    
    # Create backup of current model
    model_path = '/app/ml/inference/arrival_time_model.pth'
    scaler_path = '/app/ml/inference/scaler.pkl'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f'/app/ml/inference/backups/{timestamp}'
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup existing model if it exists
    if os.path.exists(model_path):
        shutil.copy2(model_path, f'{backup_dir}/arrival_time_model.pth')
        print(f"Backed up existing model to {backup_dir}")
    
    if os.path.exists(scaler_path):
        shutil.copy2(scaler_path, f'{backup_dir}/scaler.pkl')
        print(f"Backed up existing scaler to {backup_dir}")
    
    # Model is already in the correct location from training
    print("Model deployment completed successfully!")
    
    return "deployment_completed"

def cleanup_old_models(**context):
    """Clean up old model backups"""
    import os
    import shutil
    from datetime import datetime, timedelta
    
    print("Cleaning up old model backups...")
    
    backup_base_dir = '/app/ml/inference/backups'
    if not os.path.exists(backup_base_dir):
        print("No backup directory found, skipping cleanup")
        return
    
    # Keep backups for 30 days
    cutoff_date = datetime.now() - timedelta(days=30)
    
    removed_count = 0
    for backup_dir in os.listdir(backup_base_dir):
        backup_path = os.path.join(backup_base_dir, backup_dir)
        
        if os.path.isdir(backup_path):
            try:
                # Parse timestamp from directory name
                dir_timestamp = datetime.strptime(backup_dir, '%Y%m%d_%H%M%S')
                
                if dir_timestamp < cutoff_date:
                    shutil.rmtree(backup_path)
                    removed_count += 1
                    print(f"Removed old backup: {backup_dir}")
                    
            except ValueError:
                # Skip directories that don't match timestamp format
                continue
    
    print(f"Cleanup completed. Removed {removed_count} old backups.")

# Define tasks
generate_data_task = PythonOperator(
    task_id='generate_training_data',
    python_callable=generate_training_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_models',
    python_callable=cleanup_old_models,
    dag=dag,
)

# Health check task
health_check_task = BashOperator(
    task_id='health_check',
    bash_command='curl -f http://api-service:8000/health || exit 1',
    dag=dag,
)

# Define task dependencies
generate_data_task >> train_model_task >> validate_model_task >> deploy_model_task >> cleanup_task >> health_check_task
