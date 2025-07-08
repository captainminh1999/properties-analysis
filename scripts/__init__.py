"""
Property Analytics Tool - Scripts Package
"""

__version__ = "1.0.0"
__author__ = "Property Analytics Team"

# Import main classes for easy access (ABS and statistics-based only)
try:
    from .data_fetcher import create_realistic_property_data, fetch_abs_socioeconomic_data
    from .data_processor import PropertyDataProcessor
    from .ml_models import PropertyValuationModel
    from .monte_carlo import MonteCarloPropertySimulation
    from .visualization import PropertyVisualizationSuite
    
    __all__ = [
        'create_realistic_property_data',
        'fetch_abs_socioeconomic_data',
        'PropertyDataProcessor',
        'PropertyValuationModel',
        'MonteCarloPropertySimulation',
        'PropertyVisualizationSuite'
    ]
    
except ImportError as e:
    # Some packages may not be installed yet
    pass
