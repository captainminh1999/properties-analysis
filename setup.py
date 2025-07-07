"""
Setup script for the Property Analytics Tool.
Run this script to verify your installation and setup.
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_imports():
    """Check if all required packages can be imported"""
    print("\n🔍 Checking package imports...")
    
    packages = [
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("geopandas", "Geospatial analysis"),
        ("plotly", "Interactive visualizations"),
        ("folium", "Maps"),
        ("requests", "HTTP requests")
    ]
    
    all_good = True
    for package, description in packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (Missing)")
            all_good = False
    
    return all_good

def create_config_file():
    """Create configuration file from template"""
    print("\n⚙️ Setting up configuration...")
    
    if not os.path.exists("config.py"):
        if os.path.exists("config_template.py"):
            import shutil
            shutil.copy("config_template.py", "config.py")
            print("✅ Created config.py from template")
            print("📝 Please edit config.py and add your API keys")
        else:
            print("❌ config_template.py not found")
            return False
    else:
        print("✅ config.py already exists")
    
    return True

def check_directories():
    """Check and create required directories"""
    print("\n📁 Checking directory structure...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/outputs",
        "models",
        "notebooks",
        "scripts"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ {directory} (created)")
            except OSError:
                print(f"❌ {directory} (failed to create)")
                return False
    
    return True

def run_example():
    """Run a simple example to verify everything works"""
    print("\n🧪 Running verification test...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create simple test data
        data = pd.DataFrame({
            'price': np.random.normal(500000, 100000, 100),
            'bedrooms': np.random.choice([2, 3, 4], 100),
            'suburb': np.random.choice(['Test Suburb A', 'Test Suburb B'], 100)
        })
        
        # Basic operations
        median_price = data['price'].median()
        suburb_stats = data.groupby('suburb')['price'].median()
        
        print(f"✅ Test data created: {len(data)} records")
        print(f"✅ Median price: ${median_price:,.0f}")
        print(f"✅ Basic analysis completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🏠 Property Analytics Tool - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check/create directories
    if not check_directories():
        print("❌ Directory setup failed")
        return
    
    # Install requirements
    print("\n❓ Do you want to install/update required packages? (y/n): ", end="")
    if input().lower().startswith('y'):
        if not install_requirements():
            print("❌ Package installation failed")
            return
    
    # Check imports
    if not check_imports():
        print("\n❌ Some packages are missing. Please run:")
        print("pip install -r requirements.txt")
        return
    
    # Create config file
    create_config_file()
    
    # Run verification test
    if run_example():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit config.py and add your Domain.com.au API key")
        print("2. Open Jupyter notebook: jupyter notebook notebooks/property_analytics_main.ipynb")
        print("3. Run the example: python example_usage.py")
        print("4. Start analyzing Australian property data!")
    else:
        print("\n❌ Setup verification failed")
        print("Please check the error messages above and try again")

if __name__ == "__main__":
    main()
