# Configuration file for Property Analytics Tool
# Copy this file to config.py and fill in your actual API keys

# Domain.com.au API Configuration
DOMAIN_API_KEY = "your_domain_api_key_here"
DOMAIN_BASE_URL = "https://api.domain.com.au"

# ABS Data API Configuration (if using API instead of direct downloads)
ABS_API_BASE_URL = "https://api.data.abs.gov.au"

# Default analysis parameters
DEFAULT_SUBURBS = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]
DEFAULT_PROPERTY_TYPES = ["House", "Unit", "Townhouse"]

# Model parameters
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Monte Carlo simulation parameters
SIMULATION_YEARS = 5
SIMULATION_RUNS = 1000

# Visualization settings
PLOTLY_THEME = "plotly_white"
MAP_CENTER_LAT = -25.2744
MAP_CENTER_LNG = 133.7751
MAP_ZOOM = 4
