"""
Property Analytics Configuration Template

This file contains configuration settings for the Property Analytics Tool.

*** NO API KEYS REQUIRED ***
This tool uses only free data sources:
- ABS (Australian Bureau of Statistics) data patterns
- Realistic property data generation based on market statistics

Simply copy this file to config.py and modify settings as needed.
"""

# Analysis Configuration
ANALYSIS_SUBURBS = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide']
PROPERTY_TYPES = ['House', 'Unit', 'Townhouse']
NUM_PROPERTIES = 500  # Number of sample properties to generate

# Model Configuration
MODEL_TYPE = 'xgboost'  # or 'random_forest'
RANDOM_STATE = 42

# Simulation Configuration
SIMULATION_YEARS = 5
SIMULATION_RUNS = 1000

# Data Directories
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'
DATA_OUTPUTS_DIR = 'data/outputs'
MODELS_DIR = 'models'

# Display Settings
DISPLAY_MAX_COLUMNS = None
DISPLAY_MAX_ROWS = 100

# Export Settings
EXPORT_CSV = True
EXPORT_JSON = True
SAVE_MODEL = True

# Visualization Settings
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 8
PLOTLY_THEME = 'plotly_white'

# Geographic Settings (Australian coordinates)
DEFAULT_LATITUDE = -33.8688  # Sydney coordinates
DEFAULT_LONGITUDE = 151.2093

# Economic Parameters for Simulation
BASE_GROWTH_RATE = 0.04  # 4% annual base growth
VOLATILITY = 0.15  # 15% volatility
INTEREST_RATE_MEAN = 0.045  # 4.5% average interest rate
UNEMPLOYMENT_IMPACT = -0.02  # -2% impact per 1% unemployment increase

print("✅ Configuration loaded - No API keys required!")
print("🆓 Using free data sources: ABS statistics + realistic generation")
