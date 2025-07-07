"""
Machine learning models for property valuation and analysis.
Implements XGBoost and Random Forest models for price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import joblib
import os


class PropertyValuationModel:
    """Machine learning model for property valuation"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the valuation model
        
        Args:
            model_type: 'xgboost' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = 'price'
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed features
        """
        df_features = df.copy()
        
        # Define feature columns
        numeric_features = [
            'bedrooms', 'bathrooms', 'parking', 'land_area', 'building_area',
            'latitude', 'longitude', 'distance_to_cbd_km'
        ]
        
        categorical_features = [
            'property_type', 'suburb', 'postcode'
        ]
        
        socioeconomic_features = [
            'MEDIAN_INCOME', 'UNEMPLOYMENT_RATE', 'POPULATION', 'EDUCATION_BACHELOR_PCT'
        ]
        
        # Select available features
        available_numeric = [col for col in numeric_features if col in df_features.columns]
        available_categorical = [col for col in categorical_features if col in df_features.columns]
        available_socioeconomic = [col for col in socioeconomic_features if col in df_features.columns]
        
        # Handle missing values
        for col in available_numeric + available_socioeconomic:
            df_features[col] = df_features[col].fillna(df_features[col].median())
        
        for col in available_categorical:
            df_features[col] = df_features[col].fillna('Unknown')
        
        # Encode categorical variables
        for col in available_categorical:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_features[col].astype(str))
            else:
                # Handle unseen categories during prediction
                df_features[f'{col}_encoded'] = df_features[col].map(
                    lambda x: self.label_encoders[col].transform([str(x)])[0] 
                    if str(x) in self.label_encoders[col].classes_ else -1
                )
        
        # Create interaction features
        if 'bedrooms' in df_features.columns and 'bathrooms' in df_features.columns:
            df_features['bedroom_bathroom_ratio'] = df_features['bedrooms'] / (df_features['bathrooms'] + 1)
        
        if 'building_area' in df_features.columns and 'land_area' in df_features.columns:
            df_features['building_land_ratio'] = df_features['building_area'] / (df_features['land_area'] + 1)
        
        # Price per square meter (if building area available)
        if 'building_area' in df_features.columns and self.target_name in df_features.columns:
            df_features['price_per_sqm'] = df_features[self.target_name] / (df_features['building_area'] + 1)
        
        return df_features
    
    def train(self, 
              df: pd.DataFrame, 
              target_col: str = 'price',
              test_size: float = 0.2,
              random_state: int = 42) -> Dict:
        """
        Train the valuation model
        
        Args:
            df: Training DataFrame
            target_col: Target column name
            test_size: Test set proportion
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        self.target_name = target_col
        
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Select feature columns
        feature_cols = [col for col in df_processed.columns 
                       if col.endswith('_encoded') or col in [
                           'bedrooms', 'bathrooms', 'parking', 'land_area', 'building_area',
                           'latitude', 'longitude', 'distance_to_cbd_km',
                           'MEDIAN_INCOME', 'UNEMPLOYMENT_RATE', 'POPULATION', 'EDUCATION_BACHELOR_PCT',
                           'bedroom_bathroom_ratio', 'building_land_ratio'
                       ]]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df_processed.columns]
        self.feature_names = available_features
        
        # Prepare X and y
        X = df_processed[available_features]
        y = df_processed[target_col]
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1
            )
        else:  # random_forest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'feature_count': len(available_features)
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Select features
        X = df_processed[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.model_type = model_data['model_type']
        self.is_trained = True


def identify_overvalued_properties(df: pd.DataFrame, 
                                 model: PropertyValuationModel,
                                 threshold: float = 0.15) -> pd.DataFrame:
    """
    Identify overvalued and undervalued properties
    
    Args:
        df: DataFrame with property data
        model: Trained valuation model
        threshold: Threshold for over/under valuation (e.g., 0.15 = 15%)
        
    Returns:
        DataFrame with valuation analysis
    """
    # Make predictions
    predicted_prices = model.predict(df)
    
    # Calculate valuation metrics
    df_analysis = df.copy()
    df_analysis['predicted_price'] = predicted_prices
    df_analysis['price_difference'] = df_analysis['price'] - df_analysis['predicted_price']
    df_analysis['price_difference_pct'] = (df_analysis['price_difference'] / df_analysis['predicted_price']) * 100
    
    # Classify properties
    df_analysis['valuation_status'] = 'Fair Value'
    df_analysis.loc[df_analysis['price_difference_pct'] > threshold * 100, 'valuation_status'] = 'Overvalued'
    df_analysis.loc[df_analysis['price_difference_pct'] < -threshold * 100, 'valuation_status'] = 'Undervalued'
    
    return df_analysis


if __name__ == "__main__":
    print("Property valuation models loaded successfully!")
