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
    
    # Add ACCURATE coordinates based on real Australian suburb locations
    suburb_coords = {
        # Sydney suburbs - accurate coordinates with small spread
        'Sydney CBD': (-33.8688, 151.2093, 0.008), 'Sydney': (-33.8688, 151.2093, 0.03),
        'Bondi': (-33.8915, 151.2767, 0.006), 'Parramatta': (-33.8151, 150.9999, 0.012),
        'Manly': (-33.7969, 151.2840, 0.006), 'Chatswood': (-33.7969, 151.1816, 0.008),
        
        # Melbourne suburbs - accurate coordinates  
        'Melbourne CBD': (-37.8136, 144.9631, 0.008), 'Melbourne': (-37.8136, 144.9631, 0.03),
        'St Kilda': (-37.8676, 144.9812, 0.006), 'Richmond': (-37.8197, 144.9975, 0.008),
        'Carlton': (-37.7964, 144.9658, 0.006), 'Brighton': (-37.9061, 144.9999, 0.008),
        
        # Brisbane suburbs - accurate coordinates
        'Brisbane CBD': (-27.4698, 153.0251, 0.008), 'Brisbane': (-27.4698, 153.0251, 0.03),
        'Fortitude Valley': (-27.4560, 153.0343, 0.006), 'New Farm': (-27.4689, 153.0508, 0.006),
        'Paddington': (-27.4598, 153.0115, 0.006), 'Kangaroo Point': (-27.4797, 153.0354, 0.005),
        
        # Perth suburbs - accurate coordinates
        'Perth CBD': (-31.9505, 115.8605, 0.008), 'Perth': (-31.9505, 115.8605, 0.03),
        'Fremantle': (-32.0569, 115.7425, 0.008), 'Subiaco': (-31.9474, 115.8241, 0.006),
        'Cottesloe': (-31.9988, 115.7585, 0.005), 'Leederville': (-31.9375, 115.8394, 0.006),
        
        # Adelaide suburbs - accurate coordinates
        'Adelaide CBD': (-34.9285, 138.6007, 0.008), 'Adelaide': (-34.9285, 138.6007, 0.03),
        'Glenelg': (-34.9804, 138.5117, 0.006), 'North Adelaide': (-34.9105, 138.5928, 0.006),
        'Unley': (-34.9489, 138.6067, 0.006), 'Norwood': (-34.9234, 138.6313, 0.006)
    }
    
    latitudes = []
    longitudes = []
    for suburb in df['suburb']:
        if suburb in suburb_coords:
            base_lat, base_lon, spread = suburb_coords[suburb]
            # Add realistic small variation within suburb boundaries
            lat_var = np.random.normal(0, spread/3)  # Much tighter clustering
            lon_var = np.random.normal(0, spread/3)
            latitudes.append(base_lat + lat_var)
            longitudes.append(base_lon + lon_var)
        else:
            # More intelligent fallback based on suburb name
            if any(keyword in suburb.lower() for keyword in ['sydney', 'nsw']):
                base_lat, base_lon = -33.8688, 151.2093
            elif any(keyword in suburb.lower() for keyword in ['melbourne', 'vic']):
                base_lat, base_lon = -37.8136, 144.9631
            elif any(keyword in suburb.lower() for keyword in ['brisbane', 'qld']):
                base_lat, base_lon = -27.4698, 153.0251
            elif any(keyword in suburb.lower() for keyword in ['perth', 'wa']):
                base_lat, base_lon = -31.9505, 115.8605
            elif any(keyword in suburb.lower() for keyword in ['adelaide', 'sa']):
                base_lat, base_lon = -34.9285, 138.6007
            else:
                base_lat, base_lon = -33.8688, 151.2093  # Default to Sydney
            
            # Small variation for unknown suburbs
            latitudes.append(base_lat + np.random.normal(0, 0.02))
            longitudes.append(base_lon + np.random.normal(0, 0.02))
    
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


def get_accurate_suburb_coordinates():
    """
    Get accurate geographic coordinates for Australian suburbs.
    Based on real suburb centroids and boundaries.
    """
    # Real suburb coordinates (latitude, longitude) - more accurate than random generation
    suburb_coords = {
        # Sydney suburbs
        'Sydney CBD': {'lat': -33.8688, 'lon': 151.2093, 'spread': 0.01},
        'Bondi': {'lat': -33.8915, 'lon': 151.2767, 'spread': 0.008},
        'Parramatta': {'lat': -33.8151, 'lon': 150.9999, 'spread': 0.015},
        'Manly': {'lat': -33.7969, 'lon': 151.2840, 'spread': 0.008},
        'Chatswood': {'lat': -33.7969, 'lon': 151.1816, 'spread': 0.010},
        
        # Melbourne suburbs  
        'Melbourne CBD': {'lat': -37.8136, 'lon': 144.9631, 'spread': 0.012},
        'St Kilda': {'lat': -37.8676, 'lon': 144.9812, 'spread': 0.008},
        'Richmond': {'lat': -37.8197, 'lon': 144.9975, 'spread': 0.010},
        'Carlton': {'lat': -37.7964, 'lon': 144.9658, 'spread': 0.008},
        'Brighton': {'lat': -37.9061, 'lon': 144.9999, 'spread': 0.012},
        
        # Brisbane suburbs
        'Brisbane CBD': {'lat': -27.4698, 'lon': 153.0251, 'spread': 0.010},
        'Fortitude Valley': {'lat': -27.4560, 'lon': 153.0343, 'spread': 0.008},
        'New Farm': {'lat': -27.4689, 'lon': 153.0508, 'spread': 0.008},
        'Paddington': {'lat': -27.4598, 'lon': 153.0115, 'spread': 0.008},
        'Kangaroo Point': {'lat': -27.4797, 'lon': 153.0354, 'spread': 0.006},
        
        # Perth suburbs
        'Perth CBD': {'lat': -31.9505, 'lon': 115.8605, 'spread': 0.010},
        'Fremantle': {'lat': -32.0569, 'lon': 115.7425, 'spread': 0.010},
        'Subiaco': {'lat': -31.9474, 'lon': 115.8241, 'spread': 0.008},
        'Cottesloe': {'lat': -31.9988, 'lon': 115.7585, 'spread': 0.006},
        'Leederville': {'lat': -31.9375, 'lon': 115.8394, 'spread': 0.008},
        
        # Adelaide suburbs
        'Adelaide CBD': {'lat': -34.9285, 'lon': 138.6007, 'spread': 0.010},
        'Glenelg': {'lat': -34.9804, 'lon': 138.5117, 'spread': 0.008},
        'North Adelaide': {'lat': -34.9105, 'lon': 138.5928, 'spread': 0.008},
        'Unley': {'lat': -34.9489, 'lon': 138.6067, 'spread': 0.008},
        'Norwood': {'lat': -34.9234, 'lon': 138.6313, 'spread': 0.008},
    }
    
    return suburb_coords

def generate_realistic_suburb_coordinates(suburb, num_properties, random_state=42):
    """
    Generate realistic coordinates within suburb boundaries.
    
    Args:
        suburb (str): Suburb name
        num_properties (int): Number of coordinates to generate
        random_state (int): Random seed
        
    Returns:
        tuple: (latitudes, longitudes) arrays
    """
    np.random.seed(random_state)
    
    suburb_coords = get_accurate_suburb_coordinates()
    
    if suburb not in suburb_coords:
        # Fallback for unknown suburbs - place in Sydney area
        center_lat, center_lon, spread = -33.8688, 151.2093, 0.05
        print(f"⚠️ Unknown suburb '{suburb}', using Sydney CBD coordinates")
    else:
        coords = suburb_coords[suburb]
        center_lat, center_lon, spread = coords['lat'], coords['lon'], coords['spread']
    
    # Generate coordinates with realistic clustering within suburb
    # Use normal distribution for more realistic clustering
    latitudes = np.random.normal(center_lat, spread/3, num_properties)
    longitudes = np.random.normal(center_lon, spread/3, num_properties)
    
    # Clip to reasonable bounds (avoid ocean coordinates)
    latitudes = np.clip(latitudes, center_lat - spread, center_lat + spread)
    longitudes = np.clip(longitudes, center_lon - spread, center_lon + spread)
    
    return latitudes, longitudes

def download_abs_suburb_data():
    """
    Download real ABS suburb boundary and demographic data.
    This function provides a framework for accessing ABS statistical data.
    """
    try:
        import requests
        
        # ABS Statistical Areas Level 2 (SA2) - closest to suburbs
        abs_api_base = "https://api.data.abs.gov.au/data/"
        
        # Example: Population and dwelling data
        datasets = {
            'population': 'ABS,ERP_ASGS2021,1.0.0',
            'dwellings': 'ABS,CENSUS2021_T32_LGA,1.0.0',
            'income': 'ABS,CENSUS2021_T15_LGA,1.0.0'
        }
        
        print("🏛️ ABS Data Sources Available:")
        print("- Statistical Areas Level 2 (SA2) boundaries")
        print("- Population and dwelling counts")
        print("- Income and socio-economic indicators")
        print("- Employment and education statistics")
        
        # Note: Real ABS API calls would require specific dataset codes
        # For this demo, we'll continue with enhanced synthetic data
        
        return {
            'status': 'framework_ready',
            'datasets_available': list(datasets.keys()),
            'note': 'Real ABS API integration can be implemented here'
        }
        
    except ImportError:
        print("⚠️ requests library needed for ABS API access")
        return {'status': 'error', 'message': 'requests not available'}

# Enhanced property data generation with accurate coordinates
def create_realistic_property_data_v2(suburbs, property_types=None, num_properties=1000, random_state=42):
    """
    Enhanced version with accurate geographic coordinates and better distribution.
    
    Args:
        suburbs (list): List of suburb names
        property_types (list): Property types to include
        num_properties (int): Total number of properties to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Enhanced property dataset with accurate coordinates
    """
    if property_types is None:
        property_types = ['House', 'Unit', 'Townhouse']
    
    np.random.seed(random_state)
    
    # Calculate properties per suburb (more realistic distribution)
    properties_per_suburb = {}
    total_weight = 0
    
    # Weight suburbs by typical size/population
    suburb_weights = {
        'Sydney CBD': 2.0, 'Melbourne CBD': 2.0, 'Brisbane CBD': 1.5,
        'Perth CBD': 1.0, 'Adelaide CBD': 0.8,
        'Bondi': 1.2, 'St Kilda': 1.2, 'Fortitude Valley': 1.0,
        'Parramatta': 1.5, 'Richmond': 1.3, 'New Farm': 0.8,
        'Manly': 1.0, 'Carlton': 1.1, 'Paddington': 0.8,
        'Chatswood': 1.3, 'Brighton': 1.2, 'Kangaroo Point': 0.6,
        'Fremantle': 1.0, 'Glenelg': 0.8, 'Subiaco': 0.9,
        'Cottesloe': 0.7, 'North Adelaide': 0.8, 'Leederville': 0.7,
        'Unley': 0.8, 'Norwood': 0.8
    }
    
    for suburb in suburbs:
        weight = suburb_weights.get(suburb, 1.0)
        properties_per_suburb[suburb] = weight
        total_weight += weight
    
    # Normalize to total properties
    for suburb in suburbs:
        properties_per_suburb[suburb] = max(1, int((properties_per_suburb[suburb] / total_weight) * num_properties))
    
    all_data = []
    
    for suburb in suburbs:
        suburb_properties = properties_per_suburb[suburb]
        
        # Generate coordinates for this suburb
        lats, lons = generate_realistic_suburb_coordinates(suburb, suburb_properties, random_state)
        
        # Property types with realistic distribution
        property_types_array = np.random.choice(
            property_types, 
            suburb_properties, 
            p=[0.55, 0.35, 0.10]  # House, Unit, Townhouse
        )
        
        # Generate other property features
        suburb_data = {
            'suburb': [suburb] * suburb_properties,
            'property_type': property_types_array,
            'latitude': lats,
            'longitude': lons,
            'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], suburb_properties, 
                                       p=[0.08, 0.22, 0.35, 0.25, 0.08, 0.02]),
            'bathrooms': np.random.choice([1, 2, 3, 4, 5], suburb_properties, 
                                        p=[0.15, 0.40, 0.30, 0.12, 0.03]),
            'parking': np.random.choice([0, 1, 2, 3, 4], suburb_properties, 
                                      p=[0.10, 0.25, 0.45, 0.15, 0.05]),
            'land_size': np.random.lognormal(6.5, 0.8, suburb_properties).astype(int),
            'building_area': np.random.lognormal(5.5, 0.6, suburb_properties).astype(int)
        }
        
        # Add to overall dataset
        suburb_df = pd.DataFrame(suburb_data)
        all_data.append(suburb_df)
    
    # Combine all suburbs
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Generate realistic prices based on location and features
    combined_data = _generate_realistic_prices_v2(combined_data)
    
    # Add temporal data
    start_date = datetime.now() - timedelta(days=365)
    combined_data['date_listed'] = [
        start_date + timedelta(days=np.random.randint(0, 365)) 
        for _ in range(len(combined_data))
    ]
    combined_data['listing_type'] = 'Sale'
    
    print(f"✅ Generated {len(combined_data)} properties with accurate coordinates")
    print(f"🗺️ Geographic spread: {len(suburbs)} suburbs")
    print(f"📍 Coordinate range: Lat {combined_data['latitude'].min():.3f} to {combined_data['latitude'].max():.3f}")
    print(f"📍 Coordinate range: Lon {combined_data['longitude'].min():.3f} to {combined_data['longitude'].max():.3f}")
    
    return combined_data

def _generate_realistic_prices_v2(data):
    """Enhanced price generation with better city/suburb pricing"""
    
    # Base prices by city (more realistic AUD 2024 values)
    city_base_prices = {
        'Sydney': 1_200_000, 'Melbourne': 900_000, 'Brisbane': 700_000,
        'Perth': 600_000, 'Adelaide': 550_000
    }
    
    # Suburb multipliers (relative to city center)
    suburb_multipliers = {
        'Sydney CBD': 1.4, 'Bondi': 1.6, 'Parramatta': 0.8, 'Manly': 1.5, 'Chatswood': 1.2,
        'Melbourne CBD': 1.3, 'St Kilda': 1.2, 'Richmond': 1.1, 'Carlton': 1.3, 'Brighton': 1.4,
        'Brisbane CBD': 1.2, 'Fortitude Valley': 1.1, 'New Farm': 1.3, 'Paddington': 1.1, 'Kangaroo Point': 1.0,
        'Perth CBD': 1.1, 'Fremantle': 1.0, 'Subiaco': 1.1, 'Cottesloe': 1.3, 'Leederville': 0.9,
        'Adelaide CBD': 1.1, 'Glenelg': 0.9, 'North Adelaide': 1.2, 'Unley': 1.0, 'Norwood': 1.0
    }
    
    # Property type multipliers
    type_multipliers = {'House': 1.0, 'Unit': 0.7, 'Townhouse': 0.85}
    
    prices = []
    for _, row in data.iterrows():
        suburb = row['suburb']
        
        # Determine city from suburb
        city = None
        for city_name in city_base_prices.keys():
            if any(suburb.startswith(prefix) for prefix in [city_name, f"{city_name} "]) or suburb.endswith(f" {city_name}"):
                city = city_name
                break
        
        # Fallback city assignment
        if city is None:
            if 'Sydney' in suburb or suburb in ['Bondi', 'Parramatta', 'Manly', 'Chatswood']:
                city = 'Sydney'
            elif 'Melbourne' in suburb or suburb in ['St Kilda', 'Richmond', 'Carlton', 'Brighton']:
                city = 'Melbourne'
            elif 'Brisbane' in suburb or suburb in ['Fortitude Valley', 'New Farm', 'Paddington', 'Kangaroo Point']:
                city = 'Brisbane'
            elif 'Perth' in suburb or suburb in ['Fremantle', 'Subiaco', 'Cottesloe', 'Leederville']:
                city = 'Perth'
            else:
                city = 'Adelaide'
        
        base_price = city_base_prices[city]
        suburb_mult = suburb_multipliers.get(suburb, 1.0)
        type_mult = type_multipliers.get(row['property_type'], 1.0)
        
        # Feature adjustments
        bedroom_mult = 0.7 + (row['bedrooms'] * 0.15)  # More bedrooms = higher price
        bathroom_mult = 0.9 + (row['bathrooms'] * 0.05)
        
        # Calculate final price with some randomness
        final_price = base_price * suburb_mult * type_mult * bedroom_mult * bathroom_mult
        final_price *= np.random.normal(1.0, 0.2)  # ±20% variation
        final_price = max(200_000, int(final_price))  # Minimum price floor
        
        prices.append(final_price)
    
    data['price'] = prices
    return data


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
