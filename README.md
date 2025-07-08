# Australian Property Analytics Tool

A comprehensive Python-based property analytics solution for the Australian market, featuring realistic data generation, machine learning valuation, Monte Carlo forecasting, and interactive visualizations.

## 🏠 Features

- **Realistic Data Generation**: Generate property data based on Australian market patterns
- **ABS Data Integration**: Leverage Australian Bureau of Statistics socio-economic data
- **Advanced Analytics**: Calculate key market indicators and regional statistics
- **ML Valuation Models**: XGBoost and Random Forest models for property valuation
- **Risk Assessment**: Identify over/undervalued properties
- **Monte Carlo Simulation**: Forecast future property prices over 3-5 years
- **Interactive Visualizations**: Charts, maps, and dashboards using Plotly and Folium
- **Geospatial Analysis**: Map properties to SA2 regions using GeoPandas
- **100% Free**: No API keys or paid services required

## 📁 Project Structure

```
properties-analysis/
├── data/
│   ├── raw/          # Raw data files
│   ├── processed/    # Cleaned and processed data
│   └── outputs/      # Analysis results and reports
├── notebooks/
│   └── property_analytics_main.ipynb  # Main analysis notebook
├── models/           # Saved ML models
├── scripts/          # Core Python modules
│   ├── data_fetcher.py      # ABS data fetching and property data generation
│   ├── data_processor.py    # Data cleaning and processing
│   ├── ml_models.py         # Machine learning models
│   ├── monte_carlo.py       # Monte Carlo simulation
│   └── visualization.py     # Interactive visualizations
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/captainminh1999/properties-analysis.git
cd properties-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Analysis

```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/property_analytics_main.ipynb

# OR run the example script
python example_usage.py
```

## 📋 Prerequisites

### Required Python Packages

- **Core Data Processing**: pandas, numpy, requests
- **Geospatial Analysis**: geopandas, shapely, pyproj
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: plotly, folium, matplotlib, seaborn
- **Statistical Analysis**: scipy, statsmodels
- **Jupyter**: jupyter, ipywidgets

### No API Keys Required! 

This tool generates realistic property data based on Australian market patterns and uses publicly available ABS statistics. No paid API access needed.

## 🔧 Configuration

All configuration is handled in the notebook or script files. Key parameters include:

```python
CONFIG = {
    'ANALYSIS_SUBURBS': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'],
    'PROPERTY_TYPES': ['House', 'Unit', 'Townhouse'],
    'SIMULATION_YEARS': 5,
    'MODEL_TYPE': 'xgboost',
    'NUM_PROPERTIES': 500  # Number of properties to generate
}
DEFAULT_PROPERTY_TYPES = ["House", "Unit", "Townhouse"]

# Model parameters
SIMULATION_YEARS = 5
SIMULATION_RUNS = 1000
```

### Analysis Parameters

Customize analysis settings in the notebook:

- **Suburbs**: List of suburbs to analyze
- **Property Types**: House, Unit, Townhouse, etc.
- **Simulation Period**: 3-5 years typical
- **Model Type**: 'xgboost' or 'random_forest'

## 📊 Key Outputs

### 1. Market KPIs
- Median sale/rent prices by region
- Rental yield calculations
- Price growth trends
- Vacancy rates

### 2. Valuation Analysis
- Property valuations vs. market predictions
- Over/undervalued property identification
- Feature importance analysis

### 3. Monte Carlo Forecasting
- Future price distributions
- Confidence intervals
- Risk assessments
- Portfolio analysis

### 4. Interactive Visualizations
- Price trend charts
- Geographic heat maps
- Valuation gauges
- Simulation forecasts

## 🗺️ Geospatial Features

The tool includes advanced geospatial capabilities:

- **Coordinate Validation**: Ensures properties are within Australia
- **SA2 Mapping**: Maps properties to Statistical Area Level 2 regions
- **Distance Calculations**: Distance to CBD and amenities
- **Interactive Maps**: Folium-based property maps
- **Heat Maps**: Property density and price visualizations

## 🤖 Machine Learning Models

### XGBoost Regressor
- **Features**: Property attributes, location, socio-economic data
- **Target**: Property price
- **Validation**: Train/test split with cross-validation
- **Metrics**: R², RMSE, MAE

### Random Forest Alternative
- Robust to outliers
- Feature importance analysis
- Handles mixed data types well

### Model Features
- Property characteristics (beds, baths, parking, area)
- Location features (coordinates, distance to CBD)
- Socio-economic indicators (income, unemployment, education)
- Interaction features (bedroom/bathroom ratios)

## 📈 Monte Carlo Simulation

### Economic Variables
- **Interest Rates**: RBA cash rate variations
- **Inflation**: CPI growth expectations
- **Population Growth**: Regional population changes
- **Income Growth**: Median income projections
- **Unemployment**: Labor market conditions
- **Construction Costs**: Building cost inflation

### Simulation Parameters
- **Base Price**: Current property value
- **Simulation Period**: 5 years (configurable)
- **Number of Runs**: 1000 simulations
- **Parameter Distributions**: Normal distributions with realistic bounds

### Outputs
- Price forecast ranges
- Probability of gains/losses
- Value at Risk (VaR) calculations
- Expected annual returns

## 🎯 Use Cases

### For Investors
- **Property Screening**: Identify undervalued opportunities
- **Portfolio Optimization**: Diversification across regions
- **Risk Assessment**: Understand downside potential
- **Timing Decisions**: Market entry/exit timing

### For Analysts
- **Market Research**: Regional performance analysis
- **Trend Identification**: Emerging market patterns
- **Comparative Analysis**: Cross-suburb comparisons
- **Forecasting**: Future market scenarios

### For Developers
- **Site Selection**: Optimal development locations
- **Pricing Strategy**: Market-based pricing
- **Feasibility Studies**: Project viability assessment
- **Market Timing**: Development phase planning

## 🔄 Data Pipeline

### 1. Data Generation
```python
# Generate realistic property data
properties = create_realistic_property_data(suburbs, property_types)

# ABS socio-economic data
socio_data = fetch_abs_socioeconomic_data(suburbs)
```

### 2. Data Processing
```python
# Clean and normalize
processor = PropertyDataProcessor()
clean_data = processor.clean_property_data(properties)

# Merge datasets
merged_data = processor.merge_with_socioeconomic_data(clean_data, socio_data)
```

### 3. Analysis
```python
# Train valuation model
model = PropertyValuationModel('xgboost')
model.train(merged_data)

# Run Monte Carlo simulation
simulator = MonteCarloPropertySimulation(base_price)
results = simulator.run_simulation()
```

### 4. Visualization
```python
# Create interactive charts
viz = PropertyVisualizationSuite()
chart = viz.create_price_trend_chart(data)
map = viz.create_property_map(data)
```

## 📝 Example Usage

```python
# Initialize components (No API keys required!)
from scripts.data_fetcher import create_realistic_property_data, fetch_abs_socioeconomic_data
from scripts.ml_models import PropertyValuationModel, identify_overvalued_properties
from scripts.monte_carlo import MonteCarloPropertySimulation

# Generate realistic property data
property_data = create_realistic_property_data(
    suburbs=['Sydney', 'Melbourne'], 
    num_properties=100
)

# Fetch ABS socio-economic data
socio_data = fetch_abs_socioeconomic_data(['Sydney', 'Melbourne'])

# Train valuation model
model = PropertyValuationModel('xgboost')
model.train(property_data)

# Identify opportunities
overvalued = identify_overvalued_properties(property_data, model)

# Run price simulation
simulator = MonteCarloPropertySimulation(base_price=800000)
forecast = simulator.run_simulation()
```

## 🔒 Data Privacy

- All data fetching complies with API terms of service
- No personal information is stored or processed
- Analysis focuses on aggregated market data
- User API keys are stored locally only

## 📊 Performance Optimization

- **Vectorized Operations**: NumPy/Pandas optimizations
- **Parallel Processing**: Multi-core simulation execution
- **Memory Management**: Efficient data structures
- **Caching**: Reuse processed data where possible

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **API Rate Limits**
   - Add delays between API calls
   - Check API documentation for limits

3. **Memory Issues**
   - Reduce simulation runs
   - Process data in chunks

4. **Visualization Problems**
   - Ensure Plotly/Folium are installed
   - Check browser compatibility

### Error Handling

The tool includes comprehensive error handling:
- API connection failures
- Data validation errors
- Model training issues
- Visualization rendering problems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Australian Bureau of Statistics** for socio-economic data and market statistics
- **Open Source Community** for the excellent Python libraries used
- **Python Data Science Stack** for enabling powerful analytics

## 📞 Support

For questions or support:
- Open an issue on GitHub: https://github.com/captainminh1999/properties-analysis/issues
- Check the documentation in the notebooks
- Review the example usage patterns

## 📊 Repository

This project is hosted on GitHub: https://github.com/captainminh1999/properties-analysis

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m "Add feature"`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Issues and Bug Reports
Please use the GitHub Issues page to report bugs or request features.

---

**Happy Property Analyzing! 🏠📊**
