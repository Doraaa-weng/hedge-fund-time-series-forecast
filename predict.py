"""
Prediction and submission generation script
"""
import pandas as pd
import numpy as np
import pickle
import config
from utils import prepare_features
from feature_engineering import engineer_features
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("Time Series Forecasting - Prediction Pipeline")
    print("=" * 60)
    
    # Load test data
    print("\n[1/4] Loading test data...")
    test = pd.read_parquet(config.TEST_PATH)
    print(f"Test data shape: {test.shape}")
    
    # Feature engineering
    print("\n[2/4] Engineering features...")
    test = engineer_features(test, is_train=False)
    
    # Prepare features
    print("\n[3/4] Preparing features...")
    X_test, _ = prepare_features(test, target_col=None)
    print(f"Test features shape: {X_test.shape}")
    
    # Load model
    print("\n[4/4] Loading model and making predictions...")
    try:
        with open('lightgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except FileNotFoundError:
        print("ERROR: Model file 'lightgbm_model.pkl' not found!")
        print("Please run train.py first to train the model.")
        return
    
    # Make predictions
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test['id'].values,
        'y_target': predictions
    })
    
    # Ensure predictions are in reasonable range (clip extreme values)
    # Based on training data, target ranges from -2201 to 2314
    submission['y_target'] = submission['y_target'].clip(
        lower=-2500, 
        upper=2500
    )
    
    # Save submission
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved to '{config.SUBMISSION_PATH}'")
    print(f"Submission shape: {submission.shape}")
    print(f"\nSubmission statistics:")
    print(submission['y_target'].describe())
    
    print("\n" + "=" * 60)
    print("Prediction complete!")
    print("=" * 60)
    
    return submission


if __name__ == "__main__":
    submission = main()
