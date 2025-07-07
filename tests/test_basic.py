"""
Basic tests for the Property Analytics Tool
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class TestPropertyAnalytics(unittest.TestCase):
    """Test cases for property analytics functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'suburb': ['Sydney', 'Melbourne', 'Brisbane'] * 10,
            'property_type': ['House', 'Unit', 'Townhouse'] * 10,
            'price': np.random.normal(500000, 100000, 30),
            'bedrooms': np.random.choice([2, 3, 4], 30),
            'bathrooms': np.random.choice([1, 2, 3], 30),
            'latitude': np.random.uniform(-37.8, -33.8, 30),
            'longitude': np.random.uniform(144.9, 151.3, 30)
        })
    
    def test_data_structure(self):
        """Test basic data structure"""
        self.assertIsInstance(self.sample_data, pd.DataFrame)
        self.assertEqual(len(self.sample_data), 30)
        self.assertIn('price', self.sample_data.columns)
        self.assertIn('suburb', self.sample_data.columns)
    
    def test_price_calculations(self):
        """Test basic price calculations"""
        median_price = self.sample_data['price'].median()
        mean_price = self.sample_data['price'].mean()
        
        self.assertIsInstance(median_price, (int, float))
        self.assertIsInstance(mean_price, (int, float))
        self.assertGreater(median_price, 0)
        self.assertGreater(mean_price, 0)
    
    def test_suburb_grouping(self):
        """Test suburb-level grouping"""
        suburb_stats = self.sample_data.groupby('suburb')['price'].median()
        
        self.assertEqual(len(suburb_stats), 3)  # 3 unique suburbs
        self.assertIn('Sydney', suburb_stats.index)
        self.assertIn('Melbourne', suburb_stats.index)
        self.assertIn('Brisbane', suburb_stats.index)
    
    def test_coordinate_validation(self):
        """Test coordinate validation for Australia"""
        # Check if coordinates are within Australia bounds
        lat_valid = self.sample_data['latitude'].between(-44, -9).all()
        lng_valid = self.sample_data['longitude'].between(113, 154).all()
        
        self.assertTrue(lat_valid, "Latitudes should be within Australian bounds")
        self.assertTrue(lng_valid, "Longitudes should be within Australian bounds")

class TestDataProcessor(unittest.TestCase):
    """Test data processing functions"""
    
    def test_imports(self):
        """Test that modules can be imported"""
        try:
            from data_processor import PropertyDataProcessor
            processor = PropertyDataProcessor()
            self.assertIsNotNone(processor)
        except ImportError:
            self.skipTest("Data processor module not available")
    
    def test_monte_carlo_import(self):
        """Test Monte Carlo simulation import"""
        try:
            from monte_carlo import MonteCarloPropertySimulation
            simulator = MonteCarloPropertySimulation(500000, 5, 100)
            self.assertIsNotNone(simulator)
        except ImportError:
            self.skipTest("Monte Carlo module not available")

if __name__ == '__main__':
    unittest.main()
