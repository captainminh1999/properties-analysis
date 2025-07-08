"""
Data fetching utilities for property analytics tool.
Handles ABS data downloads and realistic property data generation.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os


class ABSDataFetcher:
    """Fetcher for Australian Bureau of Statistics data"""
    
    def __init__(self):
        self.base_url = "https://www.abs.gov.au"
        self.common_datasets = {
            "regional_population": "https://www.abs.gov.au/statistics/people/population/regional-population/latest-release",
            "housing_census": "https://www.abs.gov.au/census/find-census-data/datapacks",
            "socio_economic": "https://www.abs.gov.au/ausstats/abs@.nsf/mf/2033.0.55.001",
            "building_approvals": "https://www.abs.gov.au/statistics/industry/building-and-construction/building-approvals-australia",
            "housing_finance": "https://www.abs.gov.au/statistics/economy/finance/housing-finance-australia"
        }
    
    def download_csv_data(self, url: str, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Download CSV data from ABS
        
        Args:
            url: Direct URL to CSV file
            filename: Optional filename to save locally
            
        Returns:
            DataFrame with the data
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            if filename:
                os.makedirs('data/raw', exist_ok=True)
                with open(f'data/raw/{filename}', 'wb') as f:
                    f.write(response.content)
            
            # Try to read as CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            return df
            
        except Exception as e:
            print(f"Error downloading ABS data: {e}")
            return pd.DataFrame()
    
    def get_sa2_boundaries(self) -> str:
        """
        Get SA2 boundary shapefiles URL
        Returns URL for download
        """
        return "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files"
    
    def get_census_data(self, sa2_codes: List[str] = None) -> pd.DataFrame:
        """
        Get census data for specified SA2 areas
        For now, returns sample data based on real ABS statistics
        """
        if sa2_codes is None:
            sa2_codes = ['101011001', '201011002', '301011003', '401011004', '501011005']
        
        # Sample data based on real ABS census statistics
        data = {
            'SA2_CODE': sa2_codes,
            'SA2_NAME': ['Sydney - Harbour', 'Melbourne - Inner East', 'Brisbane - Central', 'Perth - Inner', 'Adelaide - Central'],
            'MEDIAN_INCOME': [95000, 78000, 72000, 75000, 68000],
            'UNEMPLOYMENT_RATE': [3.8, 4.5, 4.2, 4.8, 5.1],
            'POPULATION': [28000, 22000, 25000, 18000, 15000],
            'EDUCATION_BACHELOR_PCT': [68.2, 62.4, 55.9, 58.1, 52.6],
            'MEDIAN_AGE': [34, 36, 33, 35, 38],
            'FAMILY_HOUSEHOLDS_PCT': [65.2, 68.9, 62.4, 66.7, 71.2]
        }
        
        return pd.DataFrame(data)


def create_realistic_property_data(suburbs: List[str], 
                                 property_types: List[str] = None,
                                 num_properties: int = 500,
                                 random_state: int = 42) -> pd.DataFrame:
    """
    Create realistic property data based on Australian market patterns
    
    Args:
        suburbs: List of suburb names
        property_types: List of property types
        num_properties: Number of properties to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with realistic property data
    """
    if property_types is None:
        property_types = ['House', 'Unit', 'Townhouse']
    
    np.random.seed(random_state)
    
    # Create base data structure
    suburbs_repeated = np.random.choice(suburbs, num_properties)
    property_types_array = np.random.choice(property_types, num_properties, 
                                          p=[0.55, 0.35, 0.10])  # Realistic distribution
    
    # Generate property features based on realistic patterns
    data = {
        'suburb': suburbs_repeated,
        'property_type': property_types_array,
        'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], num_properties, 
                                   p=[0.08, 0.22, 0.35, 0.25, 0.08, 0.02]),
        'bathrooms': np.random.choice([1, 2, 3, 4, 5], num_properties, 
                                    p=[0.25, 0.45, 0.22, 0.06, 0.02]),
        'parking': np.random.choice([0, 1, 2, 3, 4], num_properties, 
                                  p=[0.12, 0.38, 0.35, 0.12, 0.03]),
        'land_area': np.random.lognormal(6.2, 0.8, num_properties).clip(50, 3000),  # Log-normal for realistic distribution
        'building_area': np.random.lognormal(5.0, 0.6, num_properties).clip(40, 800),
        'date_listed': pd.date_range('2023-01-01', '2024-12-31', periods=num_properties),
        'listing_type': 'Sale'
    }
    
    df = pd.DataFrame(data)
    
    # Add coordinates based on Australian major cities
    city_coords = {
        'Sydney': (-33.8688, 151.2093),
        'Melbourne': (-37.8136, 144.9631),
        'Brisbane': (-27.4698, 153.0251),
        'Perth': (-31.9505, 115.8605),
        'Adelaide': (-34.9285, 138.6007)
    }
    
    latitudes = []
    longitudes = []
    for suburb in df['suburb']:
        if suburb in city_coords:
            base_lat, base_lon = city_coords[suburb]
            # Add random variation within ~50km radius
            lat_var = np.random.normal(0, 0.2)
            lon_var = np.random.normal(0, 0.2)
            latitudes.append(base_lat + lat_var)
            longitudes.append(base_lon + lon_var)
        else:
            # Default to Sydney area if suburb not found
            latitudes.append(-33.8688 + np.random.normal(0, 0.2))
            longitudes.append(151.2093 + np.random.normal(0, 0.2))
    
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    
    # Generate realistic prices based on Australian market data (2024)
    base_prices = {
        'Sydney': 1200000,    # Higher Sydney prices
        'Melbourne': 850000,  # Melbourne market
        'Brisbane': 650000,   # Brisbane growth
        'Perth': 580000,      # Perth market
        'Adelaide': 520000    # Adelaide affordability
    }
    
    type_multipliers = {
        'House': 1.0,
        'Unit': 0.68,         # Units typically 68% of house prices
        'Townhouse': 0.82     # Townhouses between units and houses
    }
    
    prices = []
    for _, row in df.iterrows():
        base = base_prices.get(row['suburb'], 700000)  # Default price if suburb not found
        type_mult = type_multipliers[row['property_type']]
        
        # Bedroom multiplier (more bedrooms = higher price)
        bedroom_mult = 0.7 + (row['bedrooms'] * 0.15)
        
        # Bathroom multiplier
        bathroom_mult = 0.85 + (row['bathrooms'] * 0.08)
        
        # Parking multiplier
        parking_mult = 0.92 + (row['parking'] * 0.04)
        
        # Land area multiplier (diminishing returns)
        land_mult = 0.9 + (np.log(row['land_area'] / 400) * 0.1)
        
        # Building area multiplier
        building_mult = 0.8 + (row['building_area'] / 200 * 0.15)
        
        # Calculate final price
        price = base * type_mult * bedroom_mult * bathroom_mult * parking_mult * land_mult * building_mult
        
        # Add market volatility
        price *= np.random.normal(1, 0.15)  # 15% standard deviation
        
        # Ensure minimum price
        prices.append(max(price, 150000))
    
    df['price'] = prices
    
    return df


def fetch_abs_socioeconomic_data(suburbs: List[str] = None) -> pd.DataFrame:
    """
    Fetch socio-economic data from ABS for specified suburbs
    Currently returns realistic sample data based on ABS statistics
    """
    fetcher = ABSDataFetcher()
    
    if suburbs is None:
        suburbs = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide']
    
    # Map suburbs to SA2 codes (simplified mapping)
    sa2_mapping = {
        'Sydney': '101011001',
        'Melbourne': '201011002', 
        'Brisbane': '301011003',
        'Perth': '401011004',
        'Adelaide': '501011005'
    }
    
    sa2_codes = [sa2_mapping.get(suburb, f'999999{i:03d}') for i, suburb in enumerate(suburbs)]
    
    # Get census data
    census_data = fetcher.get_census_data(sa2_codes)
    
    # Map back to suburb names
    census_data['SUBURB'] = suburbs
    
    return census_data


if __name__ == "__main__":
    # Example usage
    print("ABS-based property data utilities loaded successfully!")
    
    # Test property data generation
    suburbs = ['Sydney', 'Melbourne', 'Brisbane']
    sample_data = create_realistic_property_data(suburbs, num_properties=10)
    print(f"Generated {len(sample_data)} sample properties")
    
    # Test socio-economic data
    socio_data = fetch_abs_socioeconomic_data(suburbs)
    print(f"Loaded socio-economic data for {len(socio_data)} areas")
