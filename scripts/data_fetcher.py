"""
Data fetching utilities for property analytics tool.
Handles Domain.com.au API and ABS data downloads.
"""

import requests
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os


class DomainAPIClient:
    """Client for Domain.com.au Developer API"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.domain.com.au"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'X-Api-Key': api_key,
            'Content-Type': 'application/json'
        }
    
    def search_properties(self, 
                         suburb: str = None,
                         postcode: str = None,
                         property_types: List[str] = None,
                         min_price: int = None,
                         max_price: int = None,
                         listing_type: str = "Sale",
                         page_size: int = 200) -> List[Dict]:
        """
        Search for property listings
        
        Args:
            suburb: Suburb name
            postcode: Postcode
            property_types: List of property types (House, Unit, Townhouse, etc.)
            min_price: Minimum price
            max_price: Maximum price
            listing_type: "Sale" or "Rent"
            page_size: Number of results per page
            
        Returns:
            List of property data dictionaries
        """
        endpoint = f"{self.base_url}/v1/listings/residential/_search"
        
        search_criteria = {
            "listingType": listing_type,
            "pageSize": page_size
        }
        
        if suburb:
            search_criteria["locations"] = [{"suburb": suburb}]
        if postcode:
            search_criteria["locations"] = [{"postcode": postcode}]
        if property_types:
            search_criteria["propertyTypes"] = property_types
        if min_price:
            search_criteria["minPrice"] = min_price
        if max_price:
            search_criteria["maxPrice"] = max_price
            
        try:
            response = requests.post(endpoint, 
                                   headers=self.headers, 
                                   json=search_criteria)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching property data: {e}")
            return []
    
    def get_suburb_performance(self, suburb: str, property_type: str = "House") -> Dict:
        """
        Get suburb performance statistics
        
        Args:
            suburb: Suburb name
            property_type: Property type
            
        Returns:
            Dictionary with performance metrics
        """
        endpoint = f"{self.base_url}/v1/suburbs/{suburb}/performance-statistics"
        
        params = {
            "propertyCategory": property_type,
            "chronologicalSpan": "12"  # Last 12 months
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching suburb performance: {e}")
            return {}


class ABSDataFetcher:
    """Fetcher for Australian Bureau of Statistics data"""
    
    def __init__(self):
        self.base_url = "https://www.abs.gov.au"
        self.common_datasets = {
            "regional_population": "https://www.abs.gov.au/statistics/people/population/regional-population/latest-release",
            "housing_census": "https://www.abs.gov.au/census/find-census-data/datapacks",
            "socio_economic": "https://www.abs.gov.au/ausstats/abs@.nsf/mf/2033.0.55.001"
        }
    
    def download_csv_data(self, url: str, filename: str = None) -> pd.DataFrame:
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


def fetch_property_listings(api_key: str, 
                          suburbs: List[str],
                          property_types: List[str] = None,
                          listing_type: str = "Sale") -> pd.DataFrame:
    """
    Fetch property listings for multiple suburbs
    
    Args:
        api_key: Domain API key
        suburbs: List of suburb names
        property_types: List of property types
        listing_type: "Sale" or "Rent"
        
    Returns:
        DataFrame with all property listings
    """
    if property_types is None:
        property_types = ["House", "Unit", "Townhouse"]
    
    client = DomainAPIClient(api_key)
    all_properties = []
    
    for suburb in suburbs:
        print(f"Fetching properties for {suburb}...")
        properties = client.search_properties(
            suburb=suburb,
            property_types=property_types,
            listing_type=listing_type
        )
        
        if properties:
            all_properties.extend(properties)
        
        # Rate limiting
        time.sleep(1)
    
    # Convert to DataFrame
    if all_properties:
        df = pd.json_normalize(all_properties)
        return df
    else:
        return pd.DataFrame()


def fetch_abs_socioeconomic_data() -> pd.DataFrame:
    """
    Fetch socio-economic data from ABS
    Note: This is a placeholder - actual implementation would depend on specific ABS datasets
    """
    # This would be replaced with actual ABS data URLs
    sample_data = {
        'SA2_CODE': ['101011001', '101011002', '101011003'],
        'SA2_NAME': ['Sydney - Circular Quay', 'Sydney - The Rocks', 'Sydney - CBD'],
        'MEDIAN_INCOME': [85000, 92000, 78000],
        'UNEMPLOYMENT_RATE': [4.2, 3.8, 5.1],
        'POPULATION': [2500, 1800, 12000],
        'EDUCATION_BACHELOR_PCT': [65.2, 58.4, 48.9]
    }
    
    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # Example usage
    print("Property data fetching utilities loaded successfully!")
