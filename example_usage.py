"""
Example usage script for the Property Analytics Tool.
This script demonstrates the main workflow without requiring API keys.
"""

import sys
import os
sys.path.append('scripts')

import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from data_processor import PropertyDataProcessor, calculate_property_kpis
from ml_models import PropertyValuationModel, identify_overvalued_properties
from monte_carlo import MonteCarloPropertySimulation

def create_sample_data():
    """Create sample property data for demonstration"""
    np.random.seed(42)
    
    suburbs = ['Sydney', 'Melbourne', 'Brisbane'] * 100
    property_types = np.random.choice(['House', 'Unit', 'Townhouse'], len(suburbs))
    
    data = {
        'suburb': suburbs,
        'property_type': property_types,
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], len(suburbs)),
        'bathrooms': np.random.choice([1, 2, 3, 4], len(suburbs)),
        'parking': np.random.choice([0, 1, 2, 3], len(suburbs)),
        'land_area': np.random.normal(600, 200, len(suburbs)).clip(100, 2000),
        'building_area': np.random.normal(150, 50, len(suburbs)).clip(50, 500),
        'latitude': np.random.uniform(-37.8, -33.8, len(suburbs)),
        'longitude': np.random.uniform(144.9, 151.3, len(suburbs)),
        'date_listed': pd.date_range('2023-01-01', '2024-12-31', periods=len(suburbs)),
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic prices
    base_prices = {'Sydney': 800000, 'Melbourne': 650000, 'Brisbane': 500000}
    type_multipliers = {'House': 1.0, 'Unit': 0.7, 'Townhouse': 0.85}
    
    prices = []
    for _, row in df.iterrows():
        base = base_prices[row['suburb']]
        type_mult = type_multipliers[row['property_type']]
        bedroom_mult = 1 + (row['bedrooms'] - 2) * 0.15
        area_mult = 1 + (row['building_area'] - 150) / 1000
        
        price = base * type_mult * bedroom_mult * area_mult
        price *= np.random.normal(1, 0.2)
        prices.append(max(price, 100000))
    
    df['price'] = prices
    return df

def main():
    """Main demonstration workflow"""
    print("🏠 Property Analytics Tool - Example Usage")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample property data...")
    property_data = create_sample_data()
    print(f"✅ Generated {len(property_data)} property records")
    
    # Add sample socio-economic data
    socio_data = pd.DataFrame({
        'SA2_NAME': ['Sydney', 'Melbourne', 'Brisbane'],
        'MEDIAN_INCOME': [85000, 72000, 65000],
        'UNEMPLOYMENT_RATE': [4.2, 5.1, 4.8],
        'POPULATION': [25000, 18000, 22000],
        'EDUCATION_BACHELOR_PCT': [65.2, 58.4, 48.9]
    })
    
    # Step 2: Process data
    print("\n🧹 Step 2: Processing and cleaning data...")
    processor = PropertyDataProcessor()
    clean_data = processor.clean_property_data(property_data)
    merged_data = processor.merge_with_socioeconomic_data(clean_data, socio_data)
    geo_data = processor.create_geospatial_features(merged_data)
    geo_data = processor.calculate_distance_features(geo_data)
    print(f"✅ Data processing complete. Shape: {geo_data.shape}")
    
    # Step 3: Calculate KPIs
    print("\n📈 Step 3: Calculating market KPIs...")
    kpis = calculate_property_kpis(geo_data)
    print(f"Overall median price: ${kpis['median_price']:,.0f}")
    
    suburb_stats = geo_data.groupby('suburb')['price'].median()
    print("Suburb median prices:")
    for suburb, price in suburb_stats.items():
        print(f"  {suburb}: ${price:,.0f}")
    
    # Step 4: Train ML model
    print("\n🤖 Step 4: Training valuation model...")
    model = PropertyValuationModel('random_forest')  # Use random_forest as it doesn't require xgboost
    
    try:
        metrics = model.train(geo_data, target_col='price')
        print(f"✅ Model trained successfully")
        print(f"Test R² Score: {metrics['test_r2']:.3f}")
        print(f"Test RMSE: ${metrics['test_rmse']:,.0f}")
        
        # Step 5: Identify over/undervalued properties
        print("\n🔍 Step 5: Analyzing property valuations...")
        valuation_analysis = identify_overvalued_properties(geo_data, model, threshold=0.15)
        
        valuation_summary = valuation_analysis['valuation_status'].value_counts()
        print("Valuation analysis:")
        for status, count in valuation_summary.items():
            percentage = (count / len(valuation_analysis)) * 100
            print(f"  {status}: {count} properties ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"⚠️ Model training skipped: {e}")
    
    # Step 6: Monte Carlo simulation
    print("\n🎯 Step 6: Running Monte Carlo price simulation...")
    sample_price = geo_data['price'].median()
    
    simulator = MonteCarloPropertySimulation(
        base_price=sample_price,
        simulation_years=5,
        num_simulations=500  # Reduced for faster demo
    )
    
    try:
        simulation_results = simulator.run_simulation()
        stats = simulator.get_simulation_statistics()
        
        print(f"✅ Simulation complete")
        print(f"Current price: ${sample_price:,.0f}")
        print(f"Expected price in 5 years: ${stats['final_price_mean']:,.0f}")
        print(f"Expected annual return: {stats['annual_return_pct']:.2f}%")
        print(f"Probability of gain: {stats['probability_gain']:.1%}")
        
    except Exception as e:
        print(f"⚠️ Simulation error: {e}")
    
    # Step 7: Save results
    print("\n💾 Step 7: Saving results...")
    os.makedirs('data/outputs', exist_ok=True)
    
    try:
        geo_data.to_csv('data/outputs/sample_analysis.csv', index=False)
        print("✅ Results saved to data/outputs/sample_analysis.csv")
    except Exception as e:
        print(f"⚠️ Save error: {e}")
    
    print("\n🎉 Example workflow completed successfully!")
    print("\nNext steps:")
    print("1. Get a Domain.com.au API key")
    print("2. Update config.py with your API credentials")
    print("3. Run the full Jupyter notebook analysis")
    print("4. Customize the analysis for your specific needs")

if __name__ == "__main__":
    main()
