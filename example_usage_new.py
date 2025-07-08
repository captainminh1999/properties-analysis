"""
Example usage script for the Property Analytics Tool.
This script demonstrates the main workflow using realistic property data generation.
"""

import sys
import os
sys.path.append('scripts')

import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from data_fetcher import create_realistic_property_data, fetch_abs_socioeconomic_data
from data_processor import PropertyDataProcessor, calculate_property_kpis
from ml_models import PropertyValuationModel, identify_overvalued_properties
from monte_carlo import MonteCarloPropertySimulation


def main():
    """Main workflow demonstration"""
    print("🏠 Property Analytics Tool - Example Usage")
    print("=" * 50)
    
    # Configuration
    config = {
        'suburbs': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'],
        'property_types': ['House', 'Unit', 'Townhouse'],
        'num_properties': 500,
        'random_state': 42
    }
    
    # Step 1: Generate realistic property data
    print("📊 Step 1: Generating realistic property data...")
    property_data = create_realistic_property_data(
        suburbs=config['suburbs'],
        property_types=config['property_types'],
        num_properties=config['num_properties'],
        random_state=config['random_state']
    )
    print(f"✅ Generated {len(property_data)} property records")
    
    # Step 2: Load socio-economic data
    print("\n📈 Step 2: Loading ABS socio-economic data...")
    socio_data = fetch_abs_socioeconomic_data(config['suburbs'])
    print(f"✅ Loaded socio-economic data for {len(socio_data)} regions")
    
    # Step 3: Process data
    print("\n🧹 Step 3: Processing and cleaning data...")
    processor = PropertyDataProcessor()
    clean_data = processor.clean_property_data(property_data)
    merged_data = processor.merge_with_socioeconomic_data(clean_data, socio_data)
    geo_data = processor.create_geospatial_features(merged_data)
    geo_data = processor.calculate_distance_features(geo_data)
    print(f"✅ Data processing complete. Shape: {geo_data.shape}")
    
    # Step 4: Calculate KPIs
    print("\n📈 Step 4: Calculating market KPIs...")
    kpis = calculate_property_kpis(geo_data)
    print(f"Overall median price: ${kpis['median_price']:,.0f}")
    
    suburb_stats = geo_data.groupby('suburb')['price'].median()
    print("Suburb median prices:")
    for suburb, price in suburb_stats.items():
        print(f"  {suburb}: ${price:,.0f}")
    
    # Step 5: Train ML model
    print("\n🤖 Step 5: Training valuation model...")
    model = PropertyValuationModel('random_forest')  # Use random_forest as it's more reliable
    
    try:
        metrics = model.train(geo_data, target_col='price')
        print(f"✅ Model trained successfully")
        print(f"Test R² Score: {metrics['test_r2']:.3f}")
        print(f"Test RMSE: ${metrics['test_rmse']:,.0f}")
        
        # Step 6: Identify over/undervalued properties
        print("\n🔍 Step 6: Analyzing property valuations...")
        valuation_analysis = identify_overvalued_properties(geo_data, model, threshold=0.15)
        
        valuation_summary = valuation_analysis['valuation_status'].value_counts()
        print("Valuation analysis:")
        for status, count in valuation_summary.items():
            percentage = (count / len(valuation_analysis)) * 100
            print(f"  {status}: {count} properties ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"⚠️ Model training skipped: {e}")
        valuation_analysis = geo_data.copy()
    
    # Step 7: Monte Carlo simulation
    print("\n🎯 Step 7: Running Monte Carlo price simulation...")
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
    
    # Step 8: Save results
    print("\n💾 Step 8: Saving results...")
    os.makedirs('data/outputs', exist_ok=True)
    
    try:
        geo_data.to_csv('data/outputs/sample_analysis.csv', index=False)
        if 'valuation_analysis' in locals():
            valuation_analysis.to_csv('data/outputs/valuation_analysis.csv', index=False)
        print("✅ Results saved to data/outputs/")
    except Exception as e:
        print(f"⚠️ Save error: {e}")
    
    print("\n🎉 Example workflow completed successfully!")
    print("\nKey Results:")
    print(f"📊 Analyzed {len(geo_data)} properties across {len(config['suburbs'])} cities")
    print(f"💰 Price range: ${geo_data['price'].min():,.0f} - ${geo_data['price'].max():,.0f}")
    print(f"🏠 Property types: {geo_data['property_type'].value_counts().to_dict()}")
    
    print("\nNext steps:")
    print("1. Open the Jupyter notebook for interactive analysis")
    print("2. Customize suburbs and property types for your area")
    print("3. Explore the interactive visualizations")
    print("4. Use the ML models for property valuation")


if __name__ == "__main__":
    main()
