"""
Data processing and cleaning utilities for property analytics.
Handles data normalization, merging, and geospatial operations.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime


class PropertyDataProcessor:
    """Class for processing and cleaning property data"""
    
    def __init__(self):
        self.processed_data = None
        self.merged_data = None
    
    def clean_property_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize property listings data
        
        Args:
            df: Raw property data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Standardize columns
        df_clean = self._standardize_columns(df_clean)
        
        # Parse and clean price data
        df_clean = self._clean_price_data(df_clean)
        
        # Clean address and location data
        df_clean = self._clean_location_data(df_clean)
        
        # Parse property attributes
        df_clean = self._parse_property_attributes(df_clean)
        
        # Convert dates
        df_clean = self._convert_dates(df_clean)
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill missing numerical values with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].mode().empty:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types"""
        # Common column mappings
        column_mapping = {
            'listing.listingType': 'listing_type',
            'listing.propertyType': 'property_type',
            'listing.price': 'price',
            'listing.priceDetails.price': 'price',
            'listing.suburb': 'suburb',
            'listing.postcode': 'postcode',
            'listing.bedrooms': 'bedrooms',
            'listing.bathrooms': 'bathrooms',
            'listing.carspaces': 'parking',
            'listing.landArea': 'land_area',
            'listing.buildingArea': 'building_area',
            'listing.latitude': 'latitude',
            'listing.longitude': 'longitude',
            'listing.dateAvailable': 'date_available',
            'listing.dateListed': 'date_listed'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        return df
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize price data"""
        if 'price' in df.columns:
            # Remove currency symbols and convert to numeric
            df['price'] = df['price'].astype(str).str.replace(r'[,$]', '', regex=True)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Remove outliers (properties with unrealistic prices)
            q1 = df['price'].quantile(0.01)
            q99 = df['price'].quantile(0.99)
            df = df[(df['price'] >= q1) & (df['price'] <= q99)]
        
        return df
    
    def _clean_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean address and location data"""
        # Standardize suburb names
        if 'suburb' in df.columns:
            df['suburb'] = df['suburb'].str.title().str.strip()
        
        # Validate and clean coordinates
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Remove invalid coordinates (outside Australia)
            df = df[
                (df['latitude'].between(-44, -9)) & 
                (df['longitude'].between(113, 154))
            ]
        
        return df
    
    def _parse_property_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and clean property attributes"""
        # Ensure numeric columns are properly typed
        numeric_columns = ['bedrooms', 'bathrooms', 'parking', 'land_area', 'building_area']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize property types
        if 'property_type' in df.columns:
            property_type_mapping = {
                'house': 'House',
                'unit': 'Unit',
                'apartment': 'Unit',
                'townhouse': 'Townhouse',
                'villa': 'Townhouse'
            }
            df['property_type'] = df['property_type'].str.lower().map(
                property_type_mapping
            ).fillna(df['property_type'])
        
        return df
    
    def _convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime"""
        date_columns = ['date_available', 'date_listed']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def merge_with_socioeconomic_data(self, 
                                    property_df: pd.DataFrame,
                                    socio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge property data with socio-economic data
        
        Args:
            property_df: Cleaned property data
            socio_df: Socio-economic data with SA2 codes
            
        Returns:
            Merged DataFrame
        """
        # For now, merge on suburb name (in practice, would use SA2 mapping)
        # This is a simplified approach - real implementation would use geospatial joins
        
        if 'suburb' in property_df.columns and 'SA2_NAME' in socio_df.columns:
            # Create a simple mapping based on suburb names
            merged_df = property_df.merge(
                socio_df, 
                left_on='suburb', 
                right_on='SA2_NAME', 
                how='left'
            )
        else:
            merged_df = property_df
        
        return merged_df
    
    def create_geospatial_features(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Create geospatial features from property data
        
        Args:
            df: DataFrame with latitude/longitude columns
            
        Returns:
            GeoDataFrame with Point geometries
        """
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Create Point geometries
            geometry = [
                Point(xy) for xy in zip(df['longitude'], df['latitude'])
            ]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
            return gdf
        else:
            return gpd.GeoDataFrame(df)
    
    def calculate_distance_features(self, 
                                  gdf: gpd.GeoDataFrame,
                                  poi_df: pd.DataFrame = None) -> gpd.GeoDataFrame:
        """
        Calculate distance-based features (CBD, transport, amenities)
        
        Args:
            gdf: GeoDataFrame with property locations
            poi_df: Points of interest DataFrame (optional)
            
        Returns:
            GeoDataFrame with distance features
        """
        # Define major city CBDs (simplified)
        cbd_locations = {
            'Sydney': (-33.8688, 151.2093),
            'Melbourne': (-37.8136, 144.9631),
            'Brisbane': (-27.4698, 153.0251),
            'Perth': (-31.9505, 115.8605),
            'Adelaide': (-34.9285, 138.6007)
        }
        
        # Calculate distance to nearest CBD
        def distance_to_cbd(row):
            min_distance = float('inf')
            for city, (lat, lng) in cbd_locations.items():
                cbd_point = Point(lng, lat)
                distance = row.geometry.distance(cbd_point)
                min_distance = min(min_distance, distance)
            return min_distance * 111  # Convert to approximate km
        
        if 'geometry' in gdf.columns:
            gdf['distance_to_cbd_km'] = gdf.apply(distance_to_cbd, axis=1)
        
        return gdf


def calculate_property_kpis(df: pd.DataFrame) -> Dict:
    """
    Calculate key property market indicators
    
    Args:
        df: Property DataFrame with price and location data
        
    Returns:
        Dictionary with KPI metrics
    """
    kpis = {}
    
    if 'price' in df.columns:
        kpis['median_price'] = df['price'].median()
        kpis['mean_price'] = df['price'].mean()
        kpis['price_std'] = df['price'].std()
    
    if 'suburb' in df.columns:
        kpis['suburb_price_summary'] = df.groupby('suburb')['price'].agg([
            'count', 'median', 'mean', 'std'
        ]).to_dict()
    
    if 'property_type' in df.columns:
        kpis['property_type_summary'] = df.groupby('property_type')['price'].agg([
            'count', 'median', 'mean'
        ]).to_dict()
    
    return kpis


if __name__ == "__main__":
    print("Data processing utilities loaded successfully!")
